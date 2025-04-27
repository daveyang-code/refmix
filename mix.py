#!/usr/bin/env python3
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import tempfile
import shutil
import subprocess

def separate_reference_track(reference_path, output_dir, num_stems=4):
    """
    Use Spleeter to separate the reference track into stems
    
    Parameters:
    - reference_path: Path to the reference audio file
    - output_dir: Directory where to save the separated stems
    - num_stems: Number of stems to separate into (2, 4, or 5)
    
    Returns:
    - Path to the directory containing the separated stems
    """
    print(f"Separating track into {num_stems} stems...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Spleeter
    cmd = [
        "spleeter", "separate",
        "-p", f"spleeter:{num_stems}stems",
        "-o", output_dir,
        reference_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # Spleeter creates a subdirectory with the name of the input file
    ref_name = os.path.splitext(os.path.basename(reference_path))[0]
    stems_dir = os.path.join(output_dir, ref_name)
    
    print(f"Track separated successfully. Stems saved in {stems_dir}")
    return stems_dir

def analyze_audio_file(file_path):
    """
    Analyze a single audio file to extract spectral features
    Returns a feature vector representing the audio characteristics
    """
    print(f"Analyzing audio: {file_path}")
    
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    # Spectral centroid (brightness)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Spectral bandwidth (range of frequencies)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # Spectral contrast (difference between peaks and valleys)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Zero crossing rate (noisiness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # Frequency band energies
    bands = {
        'bass': (20, 250),
        'low_mids': (250, 500),
        'mids': (500, 2000),
        'high_mids': (2000, 5000),
        'highs': (5000, 20000)
    }
    
    band_energies = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Filter the signal
        nyquist = sr / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        if high_norm > 1.0:
            high_norm = 1.0
            
        # Butterworth filter
        if low_norm < 0.001:  # For low-pass filter
            filtered = librosa.effects.preemphasis(y, coef=0.95)
        else:
            filtered = librosa.effects.preemphasis(y, coef=0.95)
            
        # Compute energy
        band_energies[band_name] = np.mean(filtered**2)
    
    # Return all features
    features = {
        'centroid': float(centroid),
        'bandwidth': float(bandwidth),
        'contrast': float(contrast),
        'rms': float(rms),
        'zcr': float(zcr),
        'bass': band_energies['bass'],
        'low_mids': band_energies['low_mids'],
        'mids': band_energies['mids'],
        'high_mids': band_energies['high_mids'],
        'highs': band_energies['highs']
    }
    
    return features, y, sr

def classify_stem(stem_name, features):
    """
    Classify a stem to determine what type of instrument/content it contains
    based on spectral features. This helps with better matching.
    """
    # Simple classification based on stem name
    stem_name = stem_name.lower()
    
    # First try to classify based on the filename
    if 'vocal' in stem_name or 'vox' in stem_name or 'voice' in stem_name:
        return 'vocals'
    elif 'drum' in stem_name or 'percussion' in stem_name or 'beat' in stem_name:
        return 'drums'
    elif 'bass' in stem_name:
        return 'bass'
    elif 'guitar' in stem_name or 'gtr' in stem_name:
        return 'guitar'
    elif 'piano' in stem_name or 'keys' in stem_name:
        return 'piano'
    elif any(x in stem_name for x in ['synth', 'pad', 'lead', 'fx']):
        return 'synth'
    
    # Use spectral features if filename doesn't give a classification
    # High zcr and centroid usually means drums/percussion
    if features['zcr'] > 0.1 and features['centroid'] > 3000:
        return 'drums'
    # Low frequency content usually means bass
    elif features['bass'] > features['mids'] * 2:
        return 'bass'
    # High mid content with moderate zcr often means vocals
    elif features['mids'] > features['bass'] and 0.01 < features['zcr'] < 0.08:
        return 'vocals'
    # Default to "other" if we can't determine
    else:
        return 'other'

def match_stems(reference_stems, user_stems):
    """
    Match user stems to reference stems based on their spectral characteristics
    Returns a dictionary mapping user stem names to reference stem names
    """
    matches = {}
    
    # Classify all stems
    ref_stem_types = {}
    for stem_name, features in reference_stems.items():
        ref_stem_types[stem_name] = classify_stem(stem_name, features[0])
    
    user_stem_types = {}
    for stem_name, features in user_stems.items():
        user_stem_types[stem_name] = classify_stem(stem_name, features[0])
    
    print("\nClassified stems:")
    print("Reference stems:")
    for stem, stem_type in ref_stem_types.items():
        print(f"  - {stem}: {stem_type}")
    print("User stems:")
    for stem, stem_type in user_stem_types.items():
        print(f"  - {stem}: {stem_type}")
    
    # Match based on stem type
    for user_stem, user_type in user_stem_types.items():
        # Find all reference stems of the same type
        matching_refs = [r for r, t in ref_stem_types.items() if t == user_type]
        
        if matching_refs:
            # If multiple matches, choose the one with the most similar spectral profile
            if len(matching_refs) > 1:
                best_match = None
                lowest_diff = float('inf')
                
                user_features = user_stems[user_stem][0]
                
                for ref_stem in matching_refs:
                    ref_features = reference_stems[ref_stem][0]
                    
                    # Calculate feature difference (simple Euclidean distance)
                    diff = 0
                    for feat in ['centroid', 'bandwidth', 'rms']:
                        if ref_features[feat] > 0:  # Avoid division by zero
                            diff += (user_features[feat] - ref_features[feat])**2 / ref_features[feat]**2
                    
                    if diff < lowest_diff:
                        lowest_diff = diff
                        best_match = ref_stem
                
                matches[user_stem] = best_match
            else:
                matches[user_stem] = matching_refs[0]
        else:
            # If no match of the same type, use 'other' or any available reference stem
            other_refs = [r for r, t in ref_stem_types.items() if t == 'other']
            if other_refs:
                matches[user_stem] = other_refs[0]
            elif ref_stem_types:
                # Pick the first one if nothing else matches
                matches[user_stem] = list(ref_stem_types.keys())[0]
            else:
                print(f"Warning: No reference stem available to match with {user_stem}")
                matches[user_stem] = None
    
    return matches

def calculate_gain_adjustments(reference_stems, user_stems, stem_matches):
    """
    Calculate gain adjustments needed for each user stem based on its matched reference stem
    """
    gains = {}
    
    for user_stem, ref_stem in stem_matches.items():
        if ref_stem is None:
            gains[user_stem] = 1.0
            continue
            
        user_features = user_stems[user_stem][0]
        ref_features = reference_stems[ref_stem][0]
        
        # Use RMS to calculate gain adjustment
        if user_features['rms'] > 0:
            # Calculate ratio between reference and user RMS
            ratio = ref_features['rms'] / user_features['rms']
            
            # Apply a square root to make the adjustment less extreme
            gains[user_stem] = np.sqrt(ratio)
        else:
            gains[user_stem] = 1.0
        
        # Frequency band balance
        band_adjustments = {}
        for band in ['bass', 'low_mids', 'mids', 'high_mids', 'highs']:
            if user_features[band] > 0:
                band_ratio = ref_features[band] / user_features[band]
                band_adjustments[band] = np.sqrt(band_ratio)  # Square root to soften effect
            else:
                band_adjustments[band] = 1.0
        
        # Combine overall gain with frequency-specific adjustments (weighted average)
        overall_weight = 0.7  # 70% weight to overall level
        band_weight = 0.3 / 5  # 30% weight distributed across 5 bands
        
        final_gain = overall_weight * gains[user_stem]
        for band, adj in band_adjustments.items():
            final_gain += band_weight * adj
            
        gains[user_stem] = final_gain
    
    # Normalize the gains to prevent overall level changes
    avg_gain = sum(gains.values()) / len(gains)
    for stem in gains:
        gains[stem] = gains[stem] / avg_gain
        
    return gains

def apply_gains_and_mix(stem_data, gains, output_path):
    """Apply calculated gains to stems and mix them down to a final track."""
    print("\nApplying gains and mixing stems:")
    
    mixed = None
    
    for stem_name, (audio, sr) in stem_data.items():
        gain = gains[stem_name]
        print(f"  - {stem_name}: gain = {gain:.2f}")
        
        # Apply gain
        adjusted_audio = audio * gain
        
        # Convert to int16 for pydub
        tmp_path = f"temp_{stem_name}.wav"
        sf.write(tmp_path, adjusted_audio, sr)
        
        # Load with pydub
        segment = AudioSegment.from_wav(tmp_path)
        
        # Add to mix
        if mixed is None:
            mixed = segment
        else:
            mixed = mixed.overlay(segment)
        
        # Clean up temp file
        os.remove(tmp_path)
    
    # Export the final mix
    mixed.export(output_path, format="wav")
    print(f"Mixed output saved to: {output_path}")

def analyze_reference_stems_directory(ref_stems_dir):
    """
    Analyze pre-existing reference stems from a directory
    """
    reference_stems = {}
    ref_stem_files = [f for f in os.listdir(ref_stems_dir) if f.endswith(('.wav', '.mp3', '.aif', '.aiff'))]
    
    print(f"\nAnalyzing {len(ref_stem_files)} pre-existing reference stems:")
    for stem_file in ref_stem_files:
        stem_path = os.path.join(ref_stems_dir, stem_file)
        stem_name = os.path.splitext(stem_file)[0]
        
        print(f"  - {stem_name}")
        features, audio, sr = analyze_audio_file(stem_path)
        reference_stems[stem_name] = (features, audio, sr)
    
    return reference_stems

def split_only(input_path, output_dir, num_stems=4):
    """
    Only split the input file with Spleeter without any mixing
    
    Parameters:
    - input_path: Path to the input audio file to be split
    - output_dir: Directory where to save the separated stems
    - num_stems: Number of stems to separate into (2, 4, or 5)
    """
    print(f"Split-only mode: Separating {input_path} into {num_stems} stems...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run Spleeter
    cmd = [
        "spleeter", "separate",
        "-p", f"spleeter:{num_stems}stems",
        "-o", output_dir,
        input_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Spleeter creates a subdirectory with the name of the input file
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        stems_dir = os.path.join(output_dir, input_name)
        
        if os.path.exists(stems_dir):
            stems = [f for f in os.listdir(stems_dir) if f.endswith('.wav')]
            print(f"Success! Split {input_path} into {len(stems)} stems:")
            for stem in stems:
                print(f"  - {stem}")
            print(f"Stems saved in: {stems_dir}")
            return stems_dir
        else:
            print(f"Error: Expected output directory {stems_dir} was not created")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running Spleeter: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def auto_mix_with_spleeter(reference_path, stems_folder, output_path, ref_stems_dir=None, num_stems=4, clean_up=True):
    """
    Main function to automatically mix stems based on a reference track,
    using Spleeter to extract stems from the reference for better analysis.
    
    Parameters:
    - reference_path: Path to the reference audio file
    - stems_folder: Folder containing user's stem audio files
    - output_path: Path to save the final mixed output
    - ref_stems_dir: Directory to save reference stems (None for temp dir)
    - num_stems: Number of stems to separate reference into (2, 4, or 5)
    - clean_up: Whether to clean up temporary files when done
    """
    # Create a temporary directory
    using_temp_dir = ref_stems_dir is None
    if using_temp_dir:
        temp_dir = tempfile.mkdtemp()
        working_dir = temp_dir
    else:
        # Use the specified directory
        if not os.path.exists(ref_stems_dir):
            os.makedirs(ref_stems_dir)
        working_dir = ref_stems_dir
    
    try:
        # Separate reference track into stems using Spleeter
        reference_stems_dir = separate_reference_track(reference_path, working_dir, num_stems)
        
        # Analyze reference stems
        reference_stems = {}
        ref_stem_files = [f for f in os.listdir(reference_stems_dir) if f.endswith('.wav')]
        
        print(f"\nAnalyzing {len(ref_stem_files)} reference stems:")
        for stem_file in ref_stem_files:
            stem_path = os.path.join(reference_stems_dir, stem_file)
            stem_name = os.path.splitext(stem_file)[0]
            
            print(f"  - {stem_name}")
            features, audio, sr = analyze_audio_file(stem_path)
            reference_stems[stem_name] = (features, audio, sr)
        
        # Collect and analyze user stems
        user_stem_files = [f for f in os.listdir(stems_folder) if f.endswith(('.wav', '.mp3', '.aif', '.aiff'))]
        
        if not user_stem_files:
            print(f"No audio files found in {stems_folder}")
            return
        
        print(f"\nAnalyzing {len(user_stem_files)} user stems:")
        user_stems = {}
        user_stem_data = {}
        
        for stem_file in user_stem_files:
            stem_path = os.path.join(stems_folder, stem_file)
            stem_name = os.path.splitext(stem_file)[0]
            
            print(f"  - {stem_name}")
            features, audio, sr = analyze_audio_file(stem_path)
            
            user_stems[stem_name] = (features, audio, sr)
            user_stem_data[stem_name] = (audio, sr)
        
        # Match user stems to reference stems
        stem_matches = match_stems(reference_stems, user_stems)
        
        print("\nStem matching results:")
        for user_stem, ref_stem in stem_matches.items():
            print(f"  - User stem '{user_stem}' matched with reference '{ref_stem}'")
        
        # Calculate gain adjustments
        gains = calculate_gain_adjustments(reference_stems, user_stems, stem_matches)
        
        print("\nCalculated gain adjustments:")
        for stem, gain in gains.items():
            print(f"  - {stem}: {gain:.2f}")
        
        # Apply gains and mix down
        apply_gains_and_mix(user_stem_data, gains, output_path)
        
        print("\nAuto-mixing complete!")
        
    finally:
        # Clean up temp files
        if using_temp_dir and clean_up:
            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up")
        elif not clean_up and using_temp_dir:
            print(f"Temporary files kept at: {temp_dir}")

def auto_mix_with_ref_stems(ref_stems_dir, user_stems_dir, output_path):
    """
    Auto-mix using pre-existing reference stems instead of splitting with Spleeter
    
    Parameters:
    - ref_stems_dir: Directory containing reference stem files
    - user_stems_dir: Directory containing user stem files
    - output_path: Path for the mixed output file
    """
    try:
        # Analyze reference stems
        reference_stems = analyze_reference_stems_directory(ref_stems_dir)
        
        if not reference_stems:
            print(f"No reference stems found in {ref_stems_dir}")
            return
        
        # Collect and analyze user stems
        user_stem_files = [f for f in os.listdir(user_stems_dir) if f.endswith(('.wav', '.mp3', '.aif', '.aiff'))]
        
        if not user_stem_files:
            print(f"No audio files found in {user_stems_dir}")
            return
        
        print(f"\nAnalyzing {len(user_stem_files)} user stems:")
        user_stems = {}
        user_stem_data = {}
        
        for stem_file in user_stem_files:
            stem_path = os.path.join(user_stems_dir, stem_file)
            stem_name = os.path.splitext(stem_file)[0]
            
            print(f"  - {stem_name}")
            features, audio, sr = analyze_audio_file(stem_path)
            
            user_stems[stem_name] = (features, audio, sr)
            user_stem_data[stem_name] = (audio, sr)
        
        # Match user stems to reference stems
        stem_matches = match_stems(reference_stems, user_stems)
        
        print("\nStem matching results:")
        for user_stem, ref_stem in stem_matches.items():
            print(f"  - User stem '{user_stem}' matched with reference '{ref_stem}'")
        
        # Calculate gain adjustments
        gains = calculate_gain_adjustments(reference_stems, user_stems, stem_matches)
        
        print("\nCalculated gain adjustments:")
        for stem, gain in gains.items():
            print(f"  - {stem}: {gain:.2f}")
        
        # Apply gains and mix down
        apply_gains_and_mix(user_stem_data, gains, output_path)
        
        print("\nAuto-mixing complete!")
        
    except Exception as e:
        print(f"An error occurred during auto-mixing: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Audio stem tools: split, analyze, and auto-mix stems')
    
    # Create a subparser for the different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Subparser for mixing with Spleeter (original functionality)
    spleeter_parser = subparsers.add_parser('spleeter', help='Use Spleeter to separate reference track into stems and mix with user stems')
    spleeter_parser.add_argument('--reference', '-r', required=True, help='Path to reference audio file')
    spleeter_parser.add_argument('--stems', '-s', required=True, help='Path to folder containing user stems')
    spleeter_parser.add_argument('--output', '-o', default='auto_mixed_output.wav', help='Output file path')
    spleeter_parser.add_argument('--ref-stems-dir', '-d', default=None, 
                        help='Directory to save reference stems (default is temporary directory)')
    spleeter_parser.add_argument('--stems-count', '-c', type=int, default=4, choices=[2, 4, 5],
                        help='Number of stems to separate reference into (2, 4, or 5)')
    spleeter_parser.add_argument('--keep-files', '-k', action='store_true',
                        help='Keep temporary files (only applies if using default temp directory)')
    
    # Subparser for mixing with pre-existing reference stems
    refstems_parser = subparsers.add_parser('refstems', help='Use pre-existing reference stems for mixing')
    refstems_parser.add_argument('--ref-stems', '-r', required=True, help='Path to folder containing reference stems')
    refstems_parser.add_argument('--user-stems', '-u', required=True, help='Path to folder containing user stems')
    refstems_parser.add_argument('--output', '-o', default='auto_mixed_output.wav', help='Output file path')
    
    # Subparser for split-only mode
    split_parser = subparsers.add_parser('split', help='Only split an audio file into stems without mixing')
    split_parser.add_argument('--input', '-i', required=True, help='Path to input audio file to be split')
    split_parser.add_argument('--output-dir', '-o', required=True, help='Directory to save the separated stems')
    split_parser.add_argument('--stems-count', '-c', type=int, default=4, choices=[2, 4, 5],
                        help='Number of stems to separate into (2, 4, or 5)')
    
    args = parser.parse_args()
    
    if args.mode == 'spleeter':
        auto_mix_with_spleeter(
            args.reference, 
            args.stems, 
            args.output, 
            args.ref_stems_dir, 
            args.stems_count,
            not args.keep_files
        )
    elif args.mode == 'refstems':
        auto_mix_with_ref_stems(
            args.ref_stems,
            args.user_stems,
            args.output
        )
    elif args.mode == 'split':
        split_only(
            args.input,
            args.output_dir,
            args.stems_count
        )
    else:
        parser.print_help()