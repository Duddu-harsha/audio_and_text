import json
import cv2
import numpy as np
import easyocr
import pytesseract
import whisper
import moviepy as mp
from transformers import pipeline
import torch
from datetime import datetime
import re
from typing import List, Dict, Any, Tuple
import warnings
from difflib import SequenceMatcher
import tempfile
import os
import config

warnings.filterwarnings("ignore")

class CompleteVideoContentAnalyzer:
    def __init__(self, whisper_model_size=config.DEFAULT_WHISPER_MODEL):
        """Initialize with text and audio analysis capabilities"""
        print("Initializing complete content analyzer...")
        
        # OCR Engines
        self.ocr_engines = {}
        
        # EasyOCR (primary)
        try:
            self.ocr_engines['easy'] = easyocr.Reader(config.OCR_LANGUAGES, gpu=torch.cuda.is_available())
            print("✓ EasyOCR initialized")
        except Exception as e:
            print(f"✗ EasyOCR failed: {e}")
        
        # Tesseract (backup)
        try:
            pytesseract.get_tesseract_version()
            self.ocr_engines['tesseract'] = True
            print("✓ Tesseract available")
        except Exception as e:
            print(f"✗ Tesseract not available: {e}")
        
        # Whisper for Speech-to-Text
        try:
            self.whisper_model = whisper.load_model(whisper_model_size)
            print(f"✓ Whisper ({whisper_model_size}) loaded")
        except Exception as e:
            print(f"✗ Whisper failed: {e}")
            self.whisper_model = None
        
        # Sentiment Analysis
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=config.SENTIMENT_MODELS[0],
                device=device
            )
            print("✓ Sentiment analyzer loaded")
        except Exception as e:
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=config.SENTIMENT_MODELS[1],
                    device=device
                )
                print("✓ Fallback sentiment analyzer loaded")
            except Exception as e2:
                print(f"✗ Sentiment analysis unavailable: {e2}")
                self.sentiment_analyzer = None
        
        # Load toxic words and false positive contexts from config
        self.toxic_words = config.TOXIC_WORDS
        self.false_positive_contexts = config.FALSE_POSITIVE_CONTEXTS
        
        print("✓ Enhanced toxicity detection initialized")
        print("Initialization complete!")
    
    def calculate_dynamic_fps(self, duration: float) -> int:
        """Calculate optimal FPS based on video duration"""
        if duration < config.VIDEO_DURATION_THRESHOLDS['short']:
            return config.FRAME_EXTRACTION_FPS['short_video']
        elif duration <= config.VIDEO_DURATION_THRESHOLDS['medium']:
            return config.FRAME_EXTRACTION_FPS['medium_video']
        else:
            return config.FRAME_EXTRACTION_FPS['long_video']
    
    def extract_audio_from_video(self, video_path: str) -> Tuple[str, Dict]:
        """Extract audio track from video"""
        try:
            print("Extracting audio from video...")
            
            # Load video
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                video.close()
                return None, {"status": "no_audio", "message": "No audio track found in video"}
            
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Extract audio with compatible parameters
            try:
                # Try with logger parameter first (newer versions)
                audio.write_audiofile(temp_audio_path, logger=None, verbose=False)
            except TypeError:
                try:
                    # Try without verbose parameter (older versions)
                    audio.write_audiofile(temp_audio_path, logger=None)
                except TypeError:
                    # Minimal parameters for maximum compatibility
                    audio.write_audiofile(temp_audio_path)
            
            audio_info = {
                "status": "success",
                "duration": audio.duration,
                "fps": getattr(audio, 'fps', None),
                "temp_path": temp_audio_path
            }
            
            # Clean up
            video.close()
            audio.close()
            
            print(f"✓ Audio extracted: {audio_info['duration']:.1f}s")
            return temp_audio_path, audio_info
            
        except Exception as e:
            print(f"✗ Audio extraction failed: {e}")
            # Try alternative method using ffmpeg directly if moviepy fails
            return self.extract_audio_with_ffmpeg(video_path)
    
    def extract_audio_with_ffmpeg(self, video_path: str) -> Tuple[str, Dict]:
        """Fallback audio extraction using ffmpeg directly"""
        try:
            import subprocess
            
            print("Trying direct ffmpeg extraction...")
            
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Use ffmpeg to extract audio
            cmd = [param.format(input=video_path, output=temp_audio_path) if '{' in param 
                   else param for param in config.FFMPEG_AUDIO_EXTRACT_CMD]
            cmd = [video_path if param == '{input}' else temp_audio_path if param == '{output}' 
                   else param for param in config.FFMPEG_AUDIO_EXTRACT_CMD]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get duration using ffprobe
                duration_cmd = [param.format(input=video_path) if '{' in param 
                               else param for param in config.FFMPEG_DURATION_CMD]
                duration_cmd = [video_path if param == '{input}' else param 
                               for param in config.FFMPEG_DURATION_CMD]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                
                try:
                    duration = float(duration_result.stdout.strip())
                except:
                    duration = 0.0
                
                audio_info = {
                    "status": "success",
                    "duration": duration,
                    "fps": config.AUDIO_SAMPLE_RATE,
                    "temp_path": temp_audio_path,
                    "method": "ffmpeg"
                }
                
                print(f"✓ Audio extracted with ffmpeg: {duration:.1f}s")
                return temp_audio_path, audio_info
            else:
                return None, {
                    "status": "error", 
                    "message": f"ffmpeg failed: {result.stderr}",
                    "method": "ffmpeg"
                }
                
        except Exception as e:
            return None, {
                "status": "error", 
                "message": f"ffmpeg extraction failed: {e}",
                "method": "ffmpeg"
            }
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Convert audio to text using Whisper"""
        if not self.whisper_model or not audio_path:
            return {
                "status": "unavailable", 
                "transcript": "", 
                "confidence": 0.0,
                "segments": []
            }
        
        try:
            print("Transcribing audio...")
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_path)
            
            transcript = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            # Calculate average confidence from segments
            confidences = []
            for segment in segments:
                if 'no_speech_prob' in segment:
                    # Convert no_speech_prob to confidence
                    speech_confidence = 1.0 - segment['no_speech_prob']
                    confidences.append(speech_confidence)
            
            avg_confidence = np.mean(confidences) if confidences else config.WHISPER_CONFIDENCE_DEFAULT
            
            print(f"✓ Transcription complete: {len(transcript)} characters")
            if transcript:
                sample = transcript[:100] + "..." if len(transcript) > 100 else transcript
                print(f"Sample: '{sample}'")
            
            return {
                "status": "success",
                "transcript": transcript,
                "confidence": round(float(avg_confidence), 3),
                "segments": [
                    {
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip(),
                        "confidence": 1.0 - seg.get("no_speech_prob", config.NO_SPEECH_PROB_DEFAULT)
                    } for seg in segments if seg.get("text", "").strip()
                ],
                "word_count": len(transcript.split()) if transcript else 0
            }
            
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            return {
                "status": "error",
                "transcript": "",
                "confidence": 0.0,
                "message": str(e),
                "segments": []
            }
    
    def extract_frames(self, video_path: str) -> tuple:
        """Extract frames with dynamic FPS"""
        print(f"Analyzing video frames...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Calculate dynamic extraction rate
        target_fps = self.calculate_dynamic_fps(duration)
        frame_interval = max(1, int(original_fps / target_fps))
        
        print(f"Video: {duration:.1f}s, {original_fps:.1f} FPS → extracting at {target_fps} FPS")
        
        frames = []
        frame_count = 0
        timestamps = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / original_fps
                
                # Resize if too large
                height, width = frame.shape[:2]
                if width > config.MAX_VIDEO_WIDTH:
                    scale = config.MAX_VIDEO_WIDTH / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                frames.append(frame)
                timestamps.append(timestamp)
                
            frame_count += 1
        
        cap.release()
        
        video_info = {
            "duration": round(duration, 2),
            "original_fps": round(original_fps, 2),
            "extraction_fps": target_fps,
            "frames_extracted": len(frames)
        }
        
        return frames, timestamps, video_info
    
    def extract_text_with_easyocr(self, frame):
        """Extract text using EasyOCR"""
        if 'easy' not in self.ocr_engines:
            return []
        
        try:
            results = self.ocr_engines['easy'].readtext(frame)
            texts = []
            
            for (bbox, text, confidence) in results:
                if text and len(text.strip()) > 0 and confidence > config.OCR_CONFIDENCE_THRESHOLD:
                    texts.append({
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "method": "easyocr"
                    })
            
            return texts
        except Exception as e:
            return []
    
    def extract_text_with_tesseract(self, frame):
        """Extract text using Tesseract"""
        if 'tesseract' not in self.ocr_engines:
            return []
        
        try:
            # Convert to grayscale and apply threshold
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            data = pytesseract.image_to_data(thresh, config=config.TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
            
            texts = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                if text and len(text) > 0 and confidence > config.TESSERACT_CONFIDENCE_THRESHOLD:
                    texts.append({
                        "text": text,
                        "confidence": confidence / 100.0,
                        "method": "tesseract"
                    })
            
            return texts
        except Exception as e:
            return []
    
    def merge_similar_texts(self, all_texts: List[str]) -> List[str]:
        """Merge similar/partial text detections into complete words"""
        if not all_texts:
            return []
        
        # Remove duplicates and sort by length (longer first)
        unique_texts = list(set(all_texts))
        unique_texts.sort(key=len, reverse=True)
        
        merged = []
        used = set()
        
        for text in unique_texts:
            if text.lower() in used:
                continue
                
            # Check if this text is contained in or very similar to existing merged text
            is_subset = False
            for merged_text in merged:
                similarity = SequenceMatcher(None, text.lower(), merged_text.lower()).ratio()
                if similarity > config.SIMILARITY_THRESHOLD or text.lower() in merged_text.lower():
                    is_subset = True
                    break
            
            if not is_subset and len(text) > config.MIN_TEXT_LENGTH:
                merged.append(text)
                used.add(text.lower())
        
        return merged
    
    def extract_text_from_video(self, frames: List[np.ndarray], timestamps: List[float]) -> Dict:
        """Extract and process text from all frames"""
        print(f"Extracting text from {len(frames)} frames...")
        
        all_raw_texts = []
        frame_details = []
        
        for i, frame in enumerate(frames):
            frame_texts = []
            
            # Try EasyOCR first
            easy_texts = self.extract_text_with_easyocr(frame)
            frame_texts.extend(easy_texts)
            
            # Try Tesseract as backup if EasyOCR finds nothing
            if not frame_texts:
                tesseract_texts = self.extract_text_with_tesseract(frame)
                frame_texts.extend(tesseract_texts)
            
            # Collect all text for this frame
            frame_text_list = [t["text"] for t in frame_texts if len(t["text"]) > 0]
            if frame_text_list:
                all_raw_texts.extend(frame_text_list)
                frame_details.append({
                    "frame": i,
                    "timestamp": round(timestamps[i], 2),
                    "texts": frame_text_list,
                    "count": len(frame_text_list)
                })
        
        # Merge and clean texts
        merged_texts = self.merge_similar_texts(all_raw_texts)
        
        # Clean up texts (remove single characters, numbers only, punctuation only)
        cleaned_texts = []
        for text in merged_texts:
            cleaned = re.sub(config.TEXT_CLEANUP_PATTERN, ' ', text).strip()
            if len(cleaned) > config.MIN_TEXT_LENGTH and not cleaned.isdigit() and any(c.isalpha() for c in cleaned):
                cleaned_texts.append(text)  # Keep original formatting
        
        print(f"✓ Visual text extraction: {len(cleaned_texts)} verified texts")
        
        return {
            "raw_detections": len(all_raw_texts),
            "verified_texts": cleaned_texts,
            "frame_details": frame_details
        }
    
    def analyze_sentiment(self, texts: List[str], source: str = "") -> Dict:
        """Analyze sentiment of detected texts"""
        if not texts or not self.sentiment_analyzer:
            return {"overall": "neutral", "score": 0.0, "individual": [], "source": source}
        
        # For audio, treat as single text block
        if source == "audio" and len(texts) == 1:
            text_to_analyze = texts[0]
        else:
            text_to_analyze = " ".join(texts[:5])  # Analyze first 5 texts combined
        
        individual_sentiments = []
        scores = []
        
        if len(text_to_analyze) > config.TEXT_MERGE_MIN_LENGTH:
            try:
                result = self.sentiment_analyzer(text_to_analyze[:config.SENTIMENT_TEXT_MAX_LENGTH])[0]
                label = result["label"].lower()
                score = float(result["score"])
                
                # Normalize labels
                if label in ['positive', 'pos']:
                    normalized_label = 'positive'
                elif label in ['negative', 'neg']:
                    normalized_label = 'negative'
                else:
                    normalized_label = 'neutral'
                
                individual_sentiments.append({
                    "text": text_to_analyze[:100] + "..." if len(text_to_analyze) > 100 else text_to_analyze,
                    "sentiment": normalized_label,
                    "confidence": score
                })
                
                # Convert to numeric for averaging
                if normalized_label == 'positive':
                    scores.append(score)
                elif normalized_label == 'negative':
                    scores.append(-score)
                else:
                    scores.append(0)
                    
            except Exception as e:
                individual_sentiments.append({
                    "text": text_to_analyze[:100] + "...",
                    "sentiment": "neutral",
                    "confidence": 0.0
                })
                scores.append(0)
        
        # Calculate overall sentiment
        if scores:
            avg_score = np.mean(scores)
            if avg_score > config.SENTIMENT_SCORE_THRESHOLDS['positive']:
                overall = "positive"
            elif avg_score < config.SENTIMENT_SCORE_THRESHOLDS['negative']:
                overall = "negative"
            else:
                overall = "neutral"
        else:
            overall = "neutral"
            avg_score = 0.0
        
        return {
            "overall": overall,
            "score": round(float(avg_score), 3),
            "individual": individual_sentiments,
            "source": source
        }
    
    def detect_toxicity_enhanced(self, texts: List[str], source: str = "") -> Dict:
        """Enhanced toxicity detection with multi-language support"""
        if not texts:
            return {
                "toxicity_level": "safe",
                "confidence": "high",
                "score": 0.0,
                "flags": [],
                "categories": {},
                "toxic_texts": [],
                "false_positives": [],
                "source": source
            }
        
        # Combine all texts for analysis
        combined_text = " ".join(texts).lower()
        
        high_confidence_toxic = []
        medium_confidence_toxic = []
        false_positives = []
        toxic_categories = {}
        toxic_text_items = []
        
        # Check all languages
        for language, categories in self.toxic_words.items():
            for category, words in categories.items():
                for word in words:
                    # Use word boundaries to avoid substring matches
                    pattern = r'\b' + re.escape(word.lower()) + r'\b'
                    matches = re.findall(pattern, combined_text)
                    
                    if matches:
                        # Check for false positives
                        is_false_positive = False
                        if word in self.false_positive_contexts:
                            for safe_word in self.false_positive_contexts[word]:
                                if safe_word in combined_text:
                                    is_false_positive = True
                                    false_positives.append({
                                        "detected_word": word,
                                        "safe_context": safe_word,
                                        "language": language,
                                        "reason": f"'{word}' found within '{safe_word}'"
                                    })
                                    break
                        
                        if not is_false_positive:
                            # Determine confidence based on severity
                            if category in ['high_severity', 'ethnic_slurs']:
                                confidence = "high"
                                high_confidence_toxic.append(word)
                            else:
                                confidence = "medium"
                                medium_confidence_toxic.append(word)
                            
                            # Add to categories
                            category_key = f"{language}_{category}"
                            if category_key not in toxic_categories:
                                toxic_categories[category_key] = []
                            
                            toxic_categories[category_key].append({
                                "word": word,
                                "confidence": confidence,
                                "language": language
                            })
                            
                            # Find which specific text contains this word
                            for text in texts:
                                if word in text.lower():
                                    toxic_text_items.append({
                                        "text": text,
                                        "toxic_word": word,
                                        "confidence": confidence,
                                        "language": language
                                    })
                                    break
        
        # Determine overall toxicity level
        all_toxic_words = list(set(high_confidence_toxic + medium_confidence_toxic))
        
        if high_confidence_toxic:
            toxicity_level = "unsafe"
            confidence = "high"
            score = 0.9
        elif medium_confidence_toxic:
            toxicity_level = "review_needed"
            confidence = "medium" 
            score = 0.5
        else:
            toxicity_level = "safe"
            confidence = "high"
            score = 0.0
        
        return {
            "toxicity_level": toxicity_level,
            "confidence": confidence,
            "score": score,
            "flags": all_toxic_words,
            "high_confidence_flags": list(set(high_confidence_toxic)),
            "medium_confidence_flags": list(set(medium_confidence_toxic)),
            "categories": toxic_categories,
            "toxic_texts": toxic_text_items,
            "false_positives": false_positives,
            "source": source
        }
    
    def combine_analysis_results(self, visual_results: Dict, audio_results: Dict) -> Dict:
        """Combine visual and audio analysis results"""
        
        # Combine toxicity assessment
        visual_toxicity = visual_results.get("toxicity_assessment", {})
        audio_toxicity = audio_results.get("toxicity_assessment", {})
        
        # Determine overall safety (most restrictive wins)
        visual_level = visual_toxicity.get("level", "safe")
        audio_level = audio_toxicity.get("level", "safe")
        
        combined_level = config.SAFETY_LEVELS[max(
            config.SAFETY_LEVELS.index(visual_level),
            config.SAFETY_LEVELS.index(audio_level)
        )]
        
        # Combine flags
        visual_flags = visual_results.get("toxicity_flags", {})
        audio_flags = audio_results.get("toxicity_flags", {})
        
        combined_flags = {
            "high_confidence": list(set(
                visual_flags.get("high_confidence", []) + 
                audio_flags.get("high_confidence", [])
            )),
            "medium_confidence": list(set(
                visual_flags.get("medium_confidence", []) + 
                audio_flags.get("medium_confidence", [])
            )),
            "false_positives": visual_flags.get("false_positives", []) + audio_flags.get("false_positives", [])
        }
        
        # Generate combined explanation
        combined_explanation = self.generate_combined_explanation(
            visual_results, audio_results, combined_level
        )
        
        return {
            "content_safety": combined_level,
            "toxicity_assessment": {
                "level": combined_level,
                "confidence": "high" if combined_level == "unsafe" else "medium" if combined_level == "review_needed" else "high",
                "sources": ["visual", "audio"],
                "visual_score": visual_toxicity.get("score", 0.0),
                "audio_score": audio_toxicity.get("score", 0.0)
            },
            "safety_explanation": combined_explanation,
            "toxicity_flags": combined_flags,
            "analysis_sources": ["visual_text", "audio_speech"]
        }
    
    def generate_combined_explanation(self, visual_results: Dict, audio_results: Dict, combined_level: str) -> str:
        """Generate explanation for combined analysis"""
        
        visual_texts = visual_results.get("detected_texts", [])
        audio_transcript = audio_results.get("transcript", "")
        
        has_visual = len(visual_texts) > 0
        has_audio = len(audio_transcript.strip()) > 0
        
        if not has_visual and not has_audio:
            return "No readable text or audible speech detected. Content safety cannot be determined."
        
        sources = []
        if has_visual:
            sources.append(f"{len(visual_texts)} visual text elements")
        if has_audio:
            word_count = len(audio_transcript.split())
            sources.append(f"audio transcript ({word_count} words)")
        
        source_desc = " and ".join(sources)
        
        if combined_level == "unsafe":
            return f"Content marked as UNSAFE based on analysis of {source_desc}. High confidence detection of offensive language requiring immediate review."
        elif combined_level == "review_needed":
            return f"Content flagged for HUMAN REVIEW based on {source_desc}. Medium confidence detection of potentially problematic language. Manual review recommended."
        else:
            return f"Content appears SAFE based on analysis of {source_desc}. No genuine toxic language identified across visual and audio content."
    
    def process_video_complete(self, video_path: str) -> Dict[str, Any]:
        """Complete video analysis pipeline - both visual and audio"""
        try:
            print(f"\n=== Complete Video Content Analysis ===")
            
            # Extract frames for visual analysis
            frames, timestamps, video_info = self.extract_frames(video_path)
            
            # Extract text from frames
            visual_text_results = self.extract_text_from_video(frames, timestamps)
            
            # Extract and analyze audio
            audio_path, audio_info = self.extract_audio_from_video(video_path)
            
            if audio_path:
                transcription_results = self.transcribe_audio(audio_path)
                # Clean up temp file
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                transcription_results = {
                    "status": "no_audio",
                    "transcript": "",
                    "confidence": 0.0,
                    "segments": []
                }
            
            # Analyze visual content
            if visual_text_results["verified_texts"]:
                visual_sentiment = self.analyze_sentiment(visual_text_results["verified_texts"], "visual")
                visual_toxicity = self.detect_toxicity_enhanced(visual_text_results["verified_texts"], "visual")
            else:
                visual_sentiment = {"overall": "neutral", "score": 0.0, "individual": [], "source": "visual"}
                visual_toxicity = {"toxicity_level": "safe", "confidence": "high", "score": 0.0, "flags": [], "source": "visual"}
            
            # Analyze audio content
            audio_transcript = transcription_results.get("transcript", "")
            if audio_transcript.strip():
                audio_texts = [audio_transcript]
                audio_sentiment = self.analyze_sentiment(audio_texts, "audio")
                audio_toxicity = self.detect_toxicity_enhanced(audio_texts, "audio")
            else:
                audio_sentiment = {"overall": "neutral", "score": 0.0, "individual": [], "source": "audio"}
                audio_toxicity = {"toxicity_level": "safe", "confidence": "high", "score": 0.0, "flags": [], "source": "audio"}
            
            # Package individual results
            visual_results = {
                "detected_texts": visual_text_results["verified_texts"],
                "text_count": len(visual_text_results["verified_texts"]),
                "sentiment_analysis": visual_sentiment,
                "toxicity_assessment": {
                    "level": visual_toxicity.get("toxicity_level", "safe"),
                    "confidence": visual_toxicity.get("confidence", "high"),
                    "score": visual_toxicity.get("score", 0.0)
                },
                "toxicity_flags": {
                    "high_confidence": visual_toxicity.get("high_confidence_flags", []),
                    "medium_confidence": visual_toxicity.get("medium_confidence_flags", []),
                    "false_positives": visual_toxicity.get("false_positives", [])
                }
            }
            
            audio_results = {
                "transcript": audio_transcript,
                "transcription_confidence": transcription_results.get("confidence", 0.0),
                "word_count": transcription_results.get("word_count", 0),
                "duration": audio_info.get("duration", 0.0),
                "segments": transcription_results.get("segments", []),
                "sentiment_analysis": audio_sentiment,
                "toxicity_assessment": {
                    "level": audio_toxicity.get("toxicity_level", "safe"),
                    "confidence": audio_toxicity.get("confidence", "high"),
                    "score": audio_toxicity.get("score", 0.0)
                },
                "toxicity_flags": {
                    "high_confidence": audio_toxicity.get("high_confidence_flags", []),
                    "medium_confidence": audio_toxicity.get("medium_confidence_flags", []),
                    "false_positives": audio_toxicity.get("false_positives", [])
                }
            }
            
            # Combine analysis results
            combined_results = self.combine_analysis_results(visual_results, audio_results)
            
            # Generate final summary
            summary = {
                "video_path": video_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "video_info": video_info,
                "content_safety": combined_results["content_safety"],
                "toxicity_assessment": combined_results["toxicity_assessment"],
                "safety_explanation": combined_results["safety_explanation"],
                "visual_analysis": visual_results,
                "audio_analysis": audio_results,
                "combined_toxicity_flags": combined_results["toxicity_flags"],
                "processing_summary": {
                    "visual_detections": visual_text_results["raw_detections"],
                    "visual_verified_texts": len(visual_text_results["verified_texts"]),
                    "audio_transcription_status": transcription_results.get("status", "unknown"),
                    "audio_confidence": transcription_results.get("confidence", 0.0),
                    "frames_with_text": len(visual_text_results["frame_details"]),
                    "engines_used": {
                        "visual": list(self.ocr_engines.keys()),
                        "audio": ["whisper"] if self.whisper_model else []
                    },
                    "languages_checked": list(self.toxic_words.keys())
                }
            }
            
            print(f"\n✓ Complete analysis finished!")
            print(f"✓ Content Safety: {summary['content_safety'].upper()}")
            print(f"✓ Visual: {len(visual_text_results['verified_texts'])} texts")
            print(f"✓ Audio: {'Transcribed' if audio_transcript.strip() else 'No speech'}")
            print(f"✓ Combined Toxicity Level: {combined_results['toxicity_assessment']['level'].upper()}")
            
            return summary
            
        except Exception as e:
            return {
                "video_path": video_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "content_safety": "unknown"
            }

def main():
    """Main execution function"""
    import sys
    import os
    
    # Get video path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        whisper_model = sys.argv[2] if len(sys.argv) > 2 else config.DEFAULT_WHISPER_MODEL
    else:
        video_files = [f for f in os.listdir('.') if f.lower().endswith(tuple(config.VIDEO_EXTENSIONS))]
        if video_files:
            video_path = video_files[0]
            whisper_model = config.DEFAULT_WHISPER_MODEL
            print(f"No video specified, using: {video_path}")
        else:
            print("Usage: python complete_analyzer.py <video_file> [whisper_model_size]")
            print("Whisper model sizes: tiny, base, small, medium, large")
            return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    try:
        # Process video with both text and audio analysis
        analyzer = CompleteVideoContentAnalyzer(whisper_model_size=whisper_model)
        results = analyzer.process_video_complete(video_path)
        
        # Generate clean JSON output
        output_filename = f"{config.OUTPUT_FILE_PREFIX}{datetime.now().strftime(config.OUTPUT_DATE_FORMAT)}.json"
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 Complete analysis saved to: {output_filename}")
        
        # Display comprehensive summary
        if results.get("content_safety"):
            print(f"\n=== COMPLETE CONTENT ANALYSIS SUMMARY ===")
            print(f"Video: {results['video_path']}")
            print(f"Duration: {results.get('video_info', {}).get('duration', 'Unknown')}s")
            print(f"Overall Safety: {results['content_safety'].upper()}")
            
            # Show toxicity assessment
            toxicity_info = results.get('toxicity_assessment', {})
            print(f"Toxicity Level: {toxicity_info.get('level', 'safe').upper()}")
            print(f"Confidence: {toxicity_info.get('confidence', 'high').upper()}")
            print(f"Sources Analyzed: {', '.join(toxicity_info.get('sources', []))}")
            
            print(f"\nExplanation: {results.get('safety_explanation', 'N/A')}")
            
            # Visual analysis summary
            visual = results.get('visual_analysis', {})
            print(f"\n--- VISUAL TEXT ANALYSIS ---")
            print(f"Texts Detected: {visual.get('text_count', 0)}")
            if visual.get('detected_texts'):
                sample_visual = ', '.join(visual['detected_texts'][:3])
                print(f"Sample: {sample_visual}{'...' if len(visual['detected_texts']) > 3 else ''}")
            print(f"Sentiment: {visual.get('sentiment_analysis', {}).get('overall', 'Unknown')}")
            print(f"Safety Level: {visual.get('toxicity_assessment', {}).get('level', 'Unknown').upper()}")
            
            # Audio analysis summary  
            audio = results.get('audio_analysis', {})
            print(f"\n--- AUDIO SPEECH ANALYSIS ---")
            print(f"Transcript Length: {audio.get('word_count', 0)} words")
            print(f"Transcription Confidence: {audio.get('transcription_confidence', 0.0):.2f}")
            if audio.get('transcript'):
                sample_audio = audio['transcript'][:100] + "..." if len(audio['transcript']) > 100 else audio['transcript']
                print(f"Sample: '{sample_audio}'")
            print(f"Sentiment: {audio.get('sentiment_analysis', {}).get('overall', 'Unknown')}")
            print(f"Safety Level: {audio.get('toxicity_assessment', {}).get('level', 'Unknown').upper()}")

            # Get combined flags
            combined_flags = results.get('combined_toxicity_flags', {})

            # Display flags
            if combined_flags.get('high_confidence'):
                print(f"\n🚨 HIGH CONFIDENCE TOXIC FLAGS: {', '.join(combined_flags['high_confidence'])}")
            if combined_flags.get('medium_confidence'):
                # Only show medium confidence flags that aren't already in high confidence
                medium_only = [flag for flag in combined_flags['medium_confidence'] if flag not in combined_flags.get('high_confidence', [])]
                if medium_only:
                    print(f"⚠️  MEDIUM CONFIDENCE FLAGS: {', '.join(medium_only)}")
            if combined_flags.get('false_positives'):
                print(f"✅ FALSE POSITIVES FILTERED: {len(combined_flags['false_positives'])}")
                
            # Processing summary
            processing = results.get('processing_summary', {})
            print(f"\n--- PROCESSING DETAILS ---")
            engines = processing.get('engines_used', {})
            print(f"Visual Engines: {', '.join(engines.get('visual', []))}")
            print(f"Audio Engines: {', '.join(engines.get('audio', []))}")
            print(f"Languages Checked: {', '.join(processing.get('languages_checked', []))}")
            print(f"Status: {results.get('status', 'completed')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()