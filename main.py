import uvicorn
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import difflib
from eng_to_ipa import convert as text_to_ipa
import language_tool_python
from difflib import SequenceMatcher
import whisper
import re
import os

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="FastAPI Server", description="FastAPI Server", version="1.0")


def preprocess_text(text):
    """Remove punctuation and normalize text for comparison"""
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def highlight_differences(reference, actual):
    """Highlight differences between reference and actual transcriptions"""
    diff = difflib.SequenceMatcher(None, reference, actual)
    highlighted_reference = []
    highlighted_actual = []

    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        ref_segment = reference[i1:i2]
        act_segment = actual[j1:j2]

        if tag == "equal":
            highlighted_reference.append(ref_segment)
            highlighted_actual.append(act_segment)
        elif tag in ("replace", "delete"):
            highlighted_reference.append(f"**{ref_segment}**")
        if tag in ("replace", "insert"):
            highlighted_actual.append(f"**{act_segment}**")

    return "".join(highlighted_reference), "".join(highlighted_actual)


def highlight_mismatched_characters(real_text, matched_text):
    """Highlight mismatched characters or words in the text."""
    highlighted_text = []
    for real_char, matched_char in zip(real_text, matched_text):
        if real_char != matched_char:
            highlighted_text.append(f"**{real_char}**")
        else:
            highlighted_text.append(real_char)
    # Append any remaining characters (if lengths differ)
    if len(real_text) > len(matched_text):
        highlighted_text.extend(
            f"**{char}**" for char in real_text[len(matched_text) :]
        )
    return "".join(highlighted_text)


def transcribe_audio_whisper(audio_path):
    """Simulated function for transcribing audio using Whisper model (replace with actual implementation)."""
    # Implement actual transcription code here
    return ["why hello there"]  # Dummy transcription for now


def evaluate_pronunciation(real_transcripts, transcribed_text):
    """Generate the result with mismatched characters highlighted."""
    # Generate matched transcripts and IPA using Whisper and eng_to_ipa

    matched_transcripts = preprocess_text(transcribed_text)

    real_transcripts_ipa = text_to_ipa(real_transcripts)
    matched_transcripts_ipa = text_to_ipa(matched_transcripts)

    # Highlight mismatched characters in text and IPA
    highlighted_text = highlight_mismatched_characters(
        real_transcripts, matched_transcripts
    )
    highlighted_ipa = highlight_differences(
        real_transcripts_ipa, matched_transcripts_ipa
    )[1]

    # Return results as a dictionary
    result = {
        "pronunciation_feedback": {
            "text_with_mistakes_highlighted": highlighted_text,
            "correct_text": real_transcripts,
            "correct_ipa": real_transcripts_ipa,
            "transcribed_ipa": highlighted_ipa,
        }
    }
    return result


@app.post("/pronunciation-evaluation")
async def pronunciation_evaluation(
    audio: UploadFile = File(...), real_transcripts: str = Form(...)
):
    """API endpoint to process pronunciation evaluation."""
    try:
        # Lưu file âm thanh tạm thời audio: UploadFile = File(...), reference_text: str = Form(...)
        audio_path = f"temp_1_{audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        transcribed_text = transcribe_audio_whisper(audio_path)[0]
        result = evaluate_pronunciation(real_transcripts, transcribed_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(audio_path)


def transcribe_audio_whisper(audio_file):
    """Sử dụng Whisper để nhận diện giọng nói."""
    model = whisper.load_model("small")  # Tải mô hình Whisper Small
    result = model.transcribe(audio_file)
    transcript = result["text"]
    segments = result["segments"]  # Lấy thời gian từng đoạn
    print(f"Transcript: {transcript}")
    return transcript, segments


# Kiểm tra
# audio_file = "./IELTS_PracticeAndEvaluation/test_data/how-are-you-doing-now-a-days.wav"
# transcript, segments = transcribe_audio_whisper(audio_file)


def calculate_grammar_score(transcript: str) -> float:
    """Tính điểm Grammar dựa trên số lỗi."""
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(transcript)
    total_words = len(transcript.split())
    grammar_errors = len(matches)

    # Tính điểm (100% nếu không có lỗi)
    grammar_score = max(0, 100 - (grammar_errors / total_words) * 100)
    return grammar_score


def calculate_vocabulary_score(transcript: str, reference_text: str) -> float:
    """Tính điểm Vocabulary dựa trên từ đúng và sai."""
    transcript_words = set(transcript.lower().split())
    reference_words = set(reference_text.lower().split())

    # Tính số từ đúng và sai
    correct_words = transcript_words.intersection(reference_words)
    vocab_score = (
        (len(correct_words) / len(reference_words)) * 100 if reference_words else 0
    )
    return vocab_score


def calculate_fluency_score(segments: list) -> float:
    """Tính điểm Fluency dựa trên số lần dừng và độ dài dừng."""
    pauses = [
        segments[i]["start"] - segments[i - 1]["end"] for i in range(1, len(segments))
    ]
    long_pauses = [pause for pause in pauses if pause > 0.5]  # Dừng dài hơn 0.5s

    # Điểm fluency dựa trên số lần dừng (ít dừng = điểm cao)
    total_pauses = len(pauses)
    fluency_score = max(
        0, 100 - (len(long_pauses) / total_pauses) * 100 if total_pauses > 0 else 0
    )
    return fluency_score


def calculate_pronunciation_score(
    transcript: str, ipa_reference: list, ipa_transcript: list
) -> float:
    """Tính điểm Pronunciation dựa trên sự khớp IPA."""
    correct_pronunciation = sum(
        1 for ref, spoken in zip(ipa_reference, ipa_transcript) if ref == spoken
    )
    total_pronunciation = len(ipa_reference)

    # Điểm phát âm dựa trên tỷ lệ khớp
    pronunciation_score = (
        (correct_pronunciation / total_pronunciation) * 100
        if total_pronunciation > 0
        else 0
    )
    return pronunciation_score


def calculate_overall_score(
    transcript: str,
    reference_text: str,
    segments: list,
    ipa_reference: list,
    ipa_transcript: list,
):
    """Tính điểm tổng hợp."""
    grammar_score = calculate_grammar_score(transcript)
    vocabulary_score = calculate_vocabulary_score(transcript, reference_text)
    fluency_score = calculate_fluency_score(segments)
    pronunciation_score = calculate_pronunciation_score(
        transcript, ipa_reference, ipa_transcript
    )

    # Tính điểm trung bình
    overall_score = (
        grammar_score + vocabulary_score + fluency_score + pronunciation_score
    ) / 4

    return {
        "grammar": grammar_score,
        "vocabulary": vocabulary_score,
        "fluency": fluency_score,
        "pronunciation": pronunciation_score,
        "overall": overall_score,
    }


def calculate_overall_score_with_ipa(
    transcript: str, reference_text: str, segments: list, ipa_errors: list
):
    """Tính điểm và thêm lỗi phát âm IPA."""
    grammar_score = calculate_grammar_score(transcript)
    vocabulary_score = calculate_vocabulary_score(transcript, reference_text)
    fluency_score = calculate_fluency_score(segments)
    pronunciation_score = 100 - len(ipa_errors)  # Giảm điểm phát âm nếu có lỗi IPA

    # Tính điểm trung bình
    overall_score = (
        grammar_score + vocabulary_score + fluency_score + pronunciation_score
    ) / 4

    return {
        "grammar": grammar_score,
        "vocabulary": vocabulary_score,
        "fluency": fluency_score,
        "pronunciation": pronunciation_score,
        "overall": overall_score,
        "ipa_errors": ipa_errors,  # Thêm chi tiết lỗi IPA
    }


def calculate_speaking_speed(transcript: str, segments: list) -> float:
    """Tính tốc độ nói theo Words Per Minute (WPM)."""
    word_count = len(transcript.split())
    total_time = (
        segments[-1]["end"] if segments else 0
    )  # Lấy thời gian kết thúc cuối cùng
    speaking_speed = (word_count / total_time) * 60 if total_time > 0 else 0  # WPM
    return round(speaking_speed, 2)


def get_grammar_details(transcript: str):
    """Phân tích lỗi ngữ pháp, trả về chi tiết lỗi."""
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(transcript)
    errors = [
        {
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "error_text": transcript[match.offset : match.offset + match.errorLength],
        }
        for match in matches
    ]
    return errors


def get_vocabulary_details(transcript: str, reference_text: str):
    """Phân tích lỗi từ vựng, trả về từ đúng và từ sai."""
    transcript_words = set(transcript.lower().split())
    reference_words = set(reference_text.lower().split())

    correct_words = transcript_words.intersection(reference_words)
    incorrect_words = transcript_words - reference_words

    return {
        "correct_words": list(correct_words),
        "incorrect_words": list(incorrect_words),
    }


def compare_ipa(transcript: str, reference_text: str) -> list:
    """
    So sánh IPA của transcript với bản tham chiếu.
    Trả về danh sách lỗi phát âm IPA.
    """
    transcript_words = transcript.split()
    reference_words = reference_text.split()

    ipa_errors = []

    # So sánh từng từ
    for ref_word, spoken_word in zip(reference_words, transcript_words):
        ref_ipa = text_to_ipa(ref_word)  # Chuyển từ tham chiếu sang IPA
        spoken_ipa = text_to_ipa(spoken_word)  # Chuyển từ transcript sang IPA

        # Tính mức độ tương đồng giữa IPA chuẩn và thực tế
        similarity = SequenceMatcher(None, ref_ipa, spoken_ipa).ratio()

        if similarity < 0.9:  # Ngưỡng sai phát âm (90% tương đồng)
            ipa_errors.append(
                {
                    "word": spoken_word,
                    "reference_ipa": ref_ipa,
                    "spoken_ipa": spoken_ipa,
                    "similarity": round(
                        similarity * 100, 2
                    ),  # Tính phần trăm tương đồng
                }
            )

    return ipa_errors


def generate_pronunciation_tips(ipa_errors: list) -> list:
    """Sinh gợi ý cải thiện phát âm từ lỗi IPA."""
    tips = []
    for error in ipa_errors:
        tips.append(
            {
                "word": error["word"],
                "tip": f"Hãy phát âm '{error['word']}' chính xác như: /{error['reference_ipa']}/",
            }
        )
    return tips


@app.post("/analyze/")
async def analyze_speaking(
    audio: UploadFile = File(...), reference_text: str = Form(...)
):
    # Lưu file âm thanh tạm thời
    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        # Gọi Whisper để lấy transcript và segments
        transcript, segments = transcribe_audio_whisper(audio_path)

        # IPA reference & transcript (giả định có hàm xử lý IPA)
        ipa_reference = text_to_ipa(reference_text)
        ipa_transcript = text_to_ipa(transcript)

        # Tính toán các tiêu chí
        grammar_details = get_grammar_details(transcript)
        vocab_details = get_vocabulary_details(transcript, reference_text)
        speaking_speed = calculate_speaking_speed(transcript, segments)
        ipa_errors = compare_ipa(transcript, reference_text)

        # Gợi ý cải thiện phát âm
        pronunciation_tips = generate_pronunciation_tips(ipa_errors)

        # Tổng hợp kết quả
        scores = calculate_overall_score_with_ipa(
            transcript,
            reference_text,
            segments,
            ipa_errors,
        )

        # Thêm tốc độ nói và chi tiết lỗi vào kết quả
        scores.update(
            {
                "speaking_speed_wpm": speaking_speed,
                "grammar_details": grammar_details,
                "vocabulary_details": vocab_details,
                "pronunciation_tips": pronunciation_tips,
            }
        )
    finally:
        # Xóa file âm thanh sau khi xử lý
        os.remove(audio_path)

    return JSONResponse(content=scores)


@app.post("/suggest_CMS/")
def suggest_CMS(WS, level, number_units):
    """Gợi ý chủ đề cho bài nói."""
    # Tính toán số chủ đề cần gợi ý
    num_suggestions = min(number_units, 5)  # Giới hạn 5 chủ đề

    # Gợi ý chủ đề
    suggestions = WS.suggest_topics(level, num_suggestions)
    return suggestions


@app.post("/check_grammar/")
def check_grammar(sentence, template_grammar, template_sample):
    """Kiểm tra ngữ pháp của câu.
    return errors: Danh sách lỗi ngữ pháp., suggestions: Gợi ý sửa lỗi.
    """
    # Kiểm tra ngữ pháp
    tool = language_tool_python.LanguageTool("en-US")

    # Kiểm tra lỗi
    matches = tool.check(sentence)

    # Lấy danh sách lỗi
    errors = [
        {
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "error_text": sentence[match.offset : match.offset + match.errorLength],
        }
        for match in matches
    ]

    # Gợi ý sửa lỗi
    suggestions = []
    for error in errors:
        suggestions.append(
            {
                "error_text": error["error_text"],
                "suggestions": error["suggestions"],
            }
        )

    return {"errors": errors, "suggestions": suggestions}


@app.post("/evaluate_writing/")
def evaluate_writing(essay, level, topic):
    """Đánh giá bài viết."""
    # Tính điểm ngữ pháp
    grammar_score = calculate_grammar_score(essay)

    # Tính điểm từ vựng
    vocab_score = calculate_vocabulary_score(essay, topic)

    # Tổng hợp điểm
    overall_score = (grammar_score + vocab_score) / 2

    return {
        "grammar_score": grammar_score,
        "vocab_score": vocab_score,
        "overall_score": overall_score,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8765,
        log_level="debug",
        reload=True,
    )
