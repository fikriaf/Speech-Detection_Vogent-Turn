#!/usr/bin/env python3
"""
File Tes untuk Vogent Turn Detection
=====================================

Vogent adalah library untuk mendeteksi kapan seseorang selesai berbicara
dalam percakapan voice AI. Library ini menggunakan:
- Audio (intonasi, jeda, nada suara)
- Teks (konteks percakapan)

Kegunaan Vogent:
1. Voice Assistant - tahu kapan user selesai bicara
2. Call Center AI - deteksi giliran bicara otomatis
3. Voice Chat Bot - respons yang lebih natural
4. Meeting Transcription - tahu kapan speaker berganti
"""

from vogent_turn import TurnDetector
import soundfile as sf
import urllib.request
import numpy as np

def contoh_1_deteksi_sederhana():
    """
    CONTOH 1: Deteksi Turn Sederhana
    --------------------------------
    Mendeteksi apakah speaker sudah selesai bicara atau masih akan melanjutkan.
    """
    print("\n" + "="*70)
    print("CONTOH 1: Deteksi Turn Sederhana")
    print("="*70)
    
    # Inisialisasi detector
    print("\n[1] Inisialisasi Turn Detector...")
    detector = TurnDetector(compile_model=False)
    
    # Download audio sample
    print("[2] Download audio sample...")
    audio_url = "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav"
    audio_file = "incomplete_number_sample.wav"
    urllib.request.urlretrieve(audio_url, audio_file)
    
    # Load audio
    print("[3] Load audio file...")
    audio, sr = sf.read(audio_file)
    
    # Konteks percakapan
    prev_line = "What is your phone number"  # Pertanyaan sebelumnya
    curr_line = "My number is 804"           # Jawaban saat ini (belum lengkap)
    
    print(f"\n[4] Konteks Percakapan:")
    print(f"    Pertanyaan: '{prev_line}'")
    print(f"    Jawaban:    '{curr_line}'")
    
    # Deteksi turn
    print("\n[5] Analisis turn endpoint...")
    result = detector.predict(
        audio,
        prev_line=prev_line,
        curr_line=curr_line,
        sample_rate=sr,
        return_probs=True
    )
    
    # Tampilkan hasil
    print("\n[6] HASIL:")
    if result['is_endpoint']:
        print("    ✅ SELESAI BICARA - Speaker sudah selesai")
    else:
        print("    ⏸️  MASIH BICARA - Speaker akan melanjutkan")
    
    print(f"\n    Confidence:")
    print(f"    - Selesai:  {result['prob_endpoint']:.1%}")
    print(f"    - Lanjut:   {result['prob_continue']:.1%}")

def contoh_2_batch_processing():
    """
    CONTOH 2: Batch Processing
    ---------------------------
    Memproses banyak audio sekaligus untuk efisiensi (cocok untuk production).
    """
    print("\n" + "="*70)
    print("CONTOH 2: Batch Processing (Efisien untuk Production)")
    print("="*70)
    
    detector = TurnDetector(compile_model=False)
    
    # Download 2 audio samples
    print("\n[1] Download audio samples...")
    samples = [
        {
            "url": "https://storage.googleapis.com/voturn-sample-recordings/incomplete_number_sample.wav",
            "file": "incomplete.wav",
            "prev": "What is your phone number",
            "curr": "My number is 804"
        },
        {
            "url": "https://storage.googleapis.com/voturn-sample-recordings/complete_number_sample.wav",
            "file": "complete.wav",
            "prev": "What is your phone number",
            "curr": "My number is 8042221111"
        }
    ]
    
    audio_batch = []
    context_batch = []
    
    for sample in samples:
        urllib.request.urlretrieve(sample["url"], sample["file"])
        audio, sr = sf.read(sample["file"])
        audio_batch.append(audio)
        context_batch.append({
            "prev_line": sample["prev"],
            "curr_line": sample["curr"]
        })
        print(f"    ✓ {sample['file']}")
    
    # Proses batch
    print("\n[2] Proses batch (lebih cepat dari satu-satu)...")
    results = detector.predict_batch(
        audio_batch,
        context_batch=context_batch,
        sample_rate=sr,
        return_probs=True
    )
    
    # Tampilkan hasil
    print("\n[3] HASIL BATCH:")
    for i, (sample, result) in enumerate(zip(samples, results)):
        status = "✅ SELESAI" if result['is_endpoint'] else "⏸️  LANJUT"
        print(f"\n    Audio {i+1}: {status}")
        print(f"    - Teks: '{sample['curr']}'")
        print(f"    - Confidence: {result['prob_endpoint']:.1%}")

def contoh_3_use_cases():
    """
    CONTOH 3: Use Cases Praktis
    ----------------------------
    Menjelaskan berbagai kegunaan Vogent dalam aplikasi nyata.
    """
    print("\n" + "="*70)
    print("CONTOH 3: Use Cases Praktis Vogent")
    print("="*70)
    
    use_cases = [
        {
            "nama": "Voice Assistant (Alexa, Siri, Google Assistant)",
            "masalah": "Kapan user selesai bicara? Jangan potong pembicaraan!",
            "solusi": "Vogent deteksi turn endpoint dengan akurat",
            "contoh": "User: 'Set alarm for...' [LANJUT] → '...7 AM tomorrow' [SELESAI]"
        },
        {
            "nama": "Call Center AI",
            "masalah": "Bot harus tahu kapan customer selesai bicara",
            "solusi": "Deteksi otomatis tanpa jeda lama yang awkward",
            "contoh": "Customer: 'I need help with...' [LANJUT] → '...my account' [SELESAI]"
        },
        {
            "nama": "Meeting Transcription",
            "masalah": "Kapan speaker berganti? Siapa yang bicara?",
            "solusi": "Deteksi turn boundaries untuk segmentasi akurat",
            "contoh": "Speaker A selesai → [ENDPOINT] → Speaker B mulai"
        },
        {
            "nama": "Voice Chat Bot",
            "masalah": "Respons terlalu cepat atau terlalu lambat",
            "solusi": "Timing yang natural dengan deteksi turn",
            "contoh": "User: 'My name is...' [LANJUT] → '...John Smith' [SELESAI] → Bot respons"
        },
        {
            "nama": "Language Learning App",
            "masalah": "Evaluasi pronunciation dan kelengkapan kalimat",
            "solusi": "Deteksi apakah learner sudah selesai bicara",
            "contoh": "Learner: 'The cat is...' [LANJUT] → '...on the table' [SELESAI]"
        }
    ]
    
    for i, uc in enumerate(use_cases, 1):
        print(f"\n{i}. {uc['nama']}")
        print(f"   Masalah: {uc['masalah']}")
        print(f"   Solusi:  {uc['solusi']}")
        print(f"   Contoh:  {uc['contoh']}")

def contoh_4_keunggulan():
    """
    CONTOH 4: Keunggulan Vogent
    ----------------------------
    Apa yang membuat Vogent lebih baik dari metode lain?
    """
    print("\n" + "="*70)
    print("CONTOH 4: Keunggulan Vogent")
    print("="*70)
    
    print("\n✨ MULTIMODAL (Audio + Text):")
    print("   - Audio: Deteksi intonasi, jeda, nada suara")
    print("   - Text: Pahami konteks percakapan")
    print("   - Hasil: Lebih akurat dari hanya audio atau text saja")
    
    print("\n⚡ CEPAT (Fast Inference):")
    print("   - Optimized dengan torch.compile")
    print("   - Batch processing untuk efisiensi")
    print("   - Cocok untuk real-time voice AI")
    
    print("\n🎯 AKURAT:")
    print("   - Model: Whisper (audio) + SmolLM (text)")
    print("   - Training: Data percakapan dengan labeled turn boundaries")
    print("   - Deteksi: Prosodic + Semantic + Contextual cues")
    
    print("\n🔧 MUDAH DIGUNAKAN:")
    print("   - API sederhana: detector.predict(audio, prev_line, curr_line)")
    print("   - Install: pip install vogent-turn")
    print("   - Production-ready: Error handling, model caching")

def main():
    """
    Main function - jalankan semua contoh
    """
    print("\n" + "="*70)
    print("TES VOGENT TURN DETECTION")
    print("="*70)
    print("\nVogent adalah library untuk deteksi giliran bicara (turn detection)")
    print("dalam percakapan voice AI.")
    print("\nLibrary ini membantu AI tahu kapan user selesai bicara,")
    print("sehingga bisa merespons dengan timing yang natural.")
    
    try:
        # Jalankan contoh-contoh
        contoh_1_deteksi_sederhana()
        contoh_2_batch_processing()
        contoh_3_use_cases()
        contoh_4_keunggulan()
        
        print("\n" + "="*70)
        print("KESIMPULAN")
        print("="*70)
        print("\nVogent berguna untuk:")
        print("  ✓ Voice assistants yang lebih natural")
        print("  ✓ Call center AI yang tidak memotong pembicaraan")
        print("  ✓ Meeting transcription yang akurat")
        print("  ✓ Voice chat bot dengan timing yang tepat")
        print("  ✓ Aplikasi voice AI lainnya yang butuh turn detection")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPastikan Anda sudah install vogent-turn:")
        print("  pip install vogent-turn")

if __name__ == "__main__":
    main()
