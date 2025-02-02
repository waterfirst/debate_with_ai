import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import random
from datetime import datetime
import html
import json
import pandas as pd
import plotly.express as px
from collections import defaultdict
import time


# AI 모델 설정
AI_MODELS = {
    "지피": {
        "name": "지피",
        "icon": "🤖",
        "model": "gpt-4o-mini",
        "color": "#00A67E",
        "system_prompt": """당신은 논리적이고 분석적인 AI로서 적극적인 토론 참여자입니다.
        다른 참여자의 의견에 대해 건설적인 반론이나 보완점을 제시하세요.
        과학적 근거와 데이터를 바탕으로 대화에 참여하되,
        필요한 경우 다른 관점도 고려하여 균형 잡힌 시각을 보여주세요.
        응답은 250~300자 이내로 작성하며, 친근한 반말로 핵심을 명확하게 전달하세요.""",
    },
    "로드": {
        "name": "로드",
        "icon": "🎩",
        "model": "claude-3-haiku-20240307",
        "color": "#7A5EA6",
        "system_prompt": """당신은 철학적이고 윤리적 관점을 중시하는 AI로서 열정적인 토론 참여자입니다.
        다른 참여자들의 의견에 대해 윤리적, 철학적 관점에서 심도 있는 분석을 제공하세요.
        특히 인간 가치와 도덕적 측면을 고려하여 토론을 더 깊이 있게 만드세요.
        때로는 도발적인 질문을 통해 토론을 활성화하되, 
        응답은 250~300자 이내로 친근한 반말로 간단명료하게 작성하세요.""",
    },
    "재민": {
        "name": "재민",
        "icon": "🚀",
        "model": "gemini-1.5-flash",
        "color": "#4285F4",
        "system_prompt": """당신은 창의적이고 혁신적인 AI로서 활발한 토론 참여자입니다.
        기존의 틀을 벗어난 새로운 시각과 미래지향적 관점을 제시하세요.
        다른 참여자들의 의견을 바탕으로 더 발전된 아이디어를 제안하고,
        때로는 파격적인 제안을 통해 토론의 지평을 넓히되,
        응답은 250~300자 이내로 친근한 반말로 명확하게 작성하세요.""",
    },
}


class AIResponseGenerator:
    """AI 응답 생성을 관리하는 클래스"""

    def __init__(self):
        try:
            self.openai_client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
            self.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_KEY"])
            genai.configure(api_key=st.secrets["GEMINI_KEY"])
            self.initialization_successful = True
        except Exception as e:
            st.error(f"API 초기화 오류: {str(e)}")
            self.initialization_successful = False

    def format_prompt(self, messages, topic):
        """대화 내용을 프롬프트로 포맷팅하여 더 적극적인 토론 유도"""
        recent_messages = messages[-3:]
        formatted_chat = "\n".join(
            [f"{msg['name']}: {msg['content']}" for msg in recent_messages]
        )

        # 토론을 더 활성화하는 프롬프트 구성
        prompt = f"""주제: {topic}

최근 대화:
{formatted_chat}

이 토론에 적극적으로 참여하여 다음 중 하나 이상을 수행해주세요:
1. 이전 의견들에 대한 건설적인 반론 제시
2. 새로운 관점이나 아이디어 제안
3. 깊이 있는 분석이나 통찰 제공
4. 다른 참여자의 의견을 발전시키거나 보완
5. 토론을 더 깊이 있게 만드는 질문 제시

200자 이내로 답변해주세요."""

        return prompt

    def generate_response(self, ai_name, topic, messages):
        """각 AI 모델별 맞춤형 응답 생성"""
        try:
            model_config = AI_MODELS[ai_name]
            prompt = self.format_prompt(messages, topic)

            if ai_name == "지피":
                response = self.openai_client.chat.completions.create(
                    model=model_config["model"],
                    messages=[
                        {"role": "system", "content": model_config["system_prompt"]},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=250,
                    temperature=0.8,  # 더 다양한 응답을 위해 temperature 증가
                )
                return response.choices[0].message.content[:250]

            elif ai_name == "로드":
                response = self.anthropic_client.messages.create(
                    model=model_config["model"],
                    max_tokens=300,
                    temperature=0.8,
                    messages=[
                        {
                            "role": "user",
                            "content": f"{model_config['system_prompt']}\n\n{prompt}",
                        }
                    ],
                )
                return response.content[0].text[:300]

            elif ai_name == "재민":
                model = genai.GenerativeModel(model_config["model"])
                response = model.generate_content(
                    f"{model_config['system_prompt']}\n\n{prompt}",
                    generation_config=genai.types.GenerationConfig(temperature=0.8),
                )
                return response.text[:250]

        except Exception as e:
            st.error(f"{ai_name} 응답 생성 오류: {str(e)}")
            return f"[시스템] {ai_name}의 응답 생성 중 오류가 발생했습니다."


def create_message_html(msg):
    """메시지 HTML 생성"""
    is_user = msg["name"] == "사용자"
    background_color = (
        "#f1c40f20" if is_user else f"{AI_MODELS[msg['name']]['color']}20"
    )
    icon = "👤" if is_user else msg.get("icon", "🤖")

    message_html = f"""
    <div class="message {'user' if is_user else 'ai'}">
        <div class="message-bubble" style="background-color: {background_color}">
            <div class="message-header">
                <strong>{msg['name']}</strong>
                <span class="ai-icon">{icon}</span>
            </div>
            <div class="message-content">
                {html.escape(msg['content'])}
            </div>
            <div class="message-time">
                {msg['time']}
            </div>
        </div>
    </div>
    """
    return message_html


def apply_styles():
    """스타일 적용"""
    st.markdown(
        """
    <style>
    
    /* 진행률 표시줄 애니메이션 */
    .stProgress > div > div > div {
        background-color: #4285f4;
        transition: width 0.3s ease;
    }
    
    /* 분석 상태 텍스트 */
    .analysis-status {
        font-size: 1.2em;
        color: #4285f4;
        text-align: center;
        margin: 20px 0;
    }
    
    /* 채팅 컨테이너 스타일 */
    .chat-container {
        max-width: 850px;
        margin: 20px auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
    }
    
    /* 메시지 스타일 */
    .message {
        display: flex;
        margin: 15px 0;
        align-items: flex-start;
    }
    
    .message.user {
        flex-direction: row-reverse;
    }
    
    .message-bubble {
        max-width: 80%;
        padding: 12px 18px;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        position: relative;
    }
    
    .user .message-bubble {
        margin-left: auto;
    }
    
    /* 텍스트 스타일 */
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        font-size: 0.95em;
    }
    
    .message-content {
        line-height: 1.5;
        font-size: 0.95em;
        word-wrap: break-word;
    }
    
    .message-time {
        font-size: 0.75em;
        color: #666;
        margin-top: 5px;
        text-align: right;
    }
    
    .ai-icon {
        font-size: 1.2em;
        margin: 0 5px;
    }
    
    /* 입력 요소 스타일 */
    [data-testid="stTextInput"] input {
        height: 50px;
        font-size: 16px;
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 0 15px;
    }
    
    [data-testid="stButton"] button {
        height: 45px;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def display_messages():
    """메시지 표시"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        st.markdown(create_message_html(msg), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def process_user_input(user_input):
    """사용자 입력 처리 및 AI 응답 생성"""
    if not user_input.strip():
        return

    # 사용자 메시지 추가
    st.session_state.messages.append(
        {
            "name": "나",
            "content": user_input,
            "time": datetime.now().strftime("%H:%M:%S"),
            "icon": "👤",
        }
    )

    # AI 응답 생성 (1-2개의 AI가 응답)
    responding_ais = random.sample(list(AI_MODELS.keys()), k=random.randint(1, 2))

    for ai_name in responding_ais:
        response = st.session_state.ai_generator.generate_response(
            ai_name, st.session_state.topic, st.session_state.messages
        )

        if response:
            st.session_state.messages.append(
                {
                    "name": ai_name,
                    "content": response,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "icon": AI_MODELS[ai_name]["icon"],
                }
            )


def save_discussion():
    """토론 내용 저장"""
    if not st.session_state.messages:
        return

    # 저장할 데이터 준비
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data = {
        "topic": st.session_state.topic,
        "timestamp": timestamp,
        "messages": st.session_state.messages,
        "metadata": {
            "participants": list(set(msg["name"] for msg in st.session_state.messages)),
            "message_count": len(st.session_state.messages),
            "duration": str(
                datetime.now()
                - datetime.strptime(st.session_state.messages[0]["time"], "%H:%M:%S")
            ),
        },
    }

    # 1. 파일 다운로드 옵션
    json_str = json.dumps(session_data, ensure_ascii=False, indent=2)
    st.download_button(
        label="💾 JSON 파일로 저장",
        data=json_str,
        file_name=f"AI_토론_{timestamp}.json",
        mime="application/json",
    )

    # 2. 복사 가능한 텍스트 형태로 표시
    with st.expander("📋 토론 내용 텍스트"):
        st.code(json_str, language="json")
        st.info("위 내용을 복사하여 나중에 불러올 수 있습니다.")

    # 3. 요약 정보 표시
    with st.expander("📊 토론 요약"):
        st.write("주제:", session_data["topic"])
        st.write("참여자:", ", ".join(session_data["metadata"]["participants"]))
        st.write("메시지 수:", session_data["metadata"]["message_count"])
        st.write("진행 시간:", session_data["metadata"]["duration"])


def save_session_to_json():
    """토론 세션을 JSON 파일로 저장"""
    if not st.session_state.messages:
        st.warning("저장할 토론 내용이 없습니다.")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data = {
        "topic": st.session_state.topic,
        "timestamp": timestamp,
        "messages": st.session_state.messages,
        "metadata": {
            "participants": list(set(msg["name"] for msg in st.session_state.messages)),
            "message_count": len(st.session_state.messages),
            "duration": (
                datetime.now()
                - datetime.strptime(st.session_state.messages[0]["time"], "%H:%M:%S")
            ).total_seconds()
            / 60,
        },
    }

    filename = f"debate_session_{timestamp}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        return filename
    except Exception as e:
        st.error(f"세션 저장 중 오류가 발생했습니다: {str(e)}")
        return False


def load_session_from_json():
    """저장된 토론 세션 불러오기"""
    uploaded_file = st.file_uploader("이전 토론 파일 선택", type=["json"])

    if uploaded_file:
        try:
            session_data = json.loads(uploaded_file.getvalue().decode("utf-8"))

            # 세션 상태 복원
            st.session_state.topic = session_data["topic"]
            st.session_state.messages = session_data["messages"]
            st.session_state.active = True

            # 메타데이터 표시
            with st.expander("불러온 세션 정보"):
                st.write(f"토론 주제: {session_data['topic']}")
                st.write(
                    f"참여자: {', '.join(session_data['metadata']['participants'])}"
                )
                st.write(f"메시지 수: {session_data['metadata']['message_count']}")
                st.write(f"토론 시간: {session_data['metadata']['duration']:.1f}분")

            return True
        except Exception as e:
            st.error(f"세션 불러오기 중 오류가 발생했습니다: {str(e)}")
            return False


def analyze_debate_participation():
    """토론 참여 분석 및 시각화"""
    if not st.session_state.messages:
        st.warning("분석할 토론 내용이 없습니다.")
        return
        # 진행률 표시줄 생성
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 분석 결과 표시
    with st.container():
        st.markdown(f"### 📊 토론 분석 결과: {st.session_state.last_topic}")

        # 기존 분석 코드
        participation_data = defaultdict(
            lambda: {
                "message_count": 0,
                "total_chars": 0,
                "response_times": [],
                "interactions": defaultdict(int),
            }
        )

    # 메시지 분석
    for i, msg in enumerate(st.session_state.messages):
        name = msg["name"]
        content = msg["content"]

        # 기본 통계 수집
        participation_data[name]["message_count"] += 1
        participation_data[name]["total_chars"] += len(content)

        # 응답 시간 분석
        if i > 0:
            prev_time = datetime.strptime(
                st.session_state.messages[i - 1]["time"], "%H:%M:%S"
            )
            curr_time = datetime.strptime(msg["time"], "%H:%M:%S")
            time_diff = (curr_time - prev_time).total_seconds()
            participation_data[name]["response_times"].append(time_diff)

        # 대화 상호작용 패턴 분석
        if i > 0:
            prev_speaker = st.session_state.messages[i - 1]["name"]
            participation_data[name]["interactions"][prev_speaker] += 1

    # 분석 결과 시각화
    st.markdown("### 📊 토론 참여 분석")

    # 1. 발언 횟수 시각화
    message_counts = {
        name: data["message_count"] for name, data in participation_data.items()
    }

    fig1 = px.bar(
        x=list(message_counts.keys()),
        y=list(message_counts.values()),
        title="참여자별 발언 횟수",
        labels={"x": "참여자", "y": "발언 횟수"},
        color=list(message_counts.keys()),
        color_discrete_map={
            "사용자": "#FFD700",
            **{name: AI_MODELS[name]["color"] for name in AI_MODELS},
        },
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. 평균 메시지 길이 시각화
    avg_lengths = {
        name: data["total_chars"] / data["message_count"]
        for name, data in participation_data.items()
    }

    fig2 = px.bar(
        x=list(avg_lengths.keys()),
        y=list(avg_lengths.values()),
        title="참여자별 평균 발언 길이",
        labels={"x": "참여자", "y": "평균 글자 수"},
        color=list(avg_lengths.keys()),
        color_discrete_map={
            "사용자": "#FFD700",
            **{name: AI_MODELS[name]["color"] for name in AI_MODELS},
        },
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. 시간대별 참여 패턴
    timeline_data = [
        {"time": msg["time"], "participant": msg["name"]}
        for msg in st.session_state.messages
    ]

    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        fig3 = px.scatter(
            timeline_df,
            x="time",
            y="participant",
            title="시간대별 참여 패턴",
            labels={"time": "시간", "participant": "참여자"},
            color="participant",
            color_discrete_map={
                "사용자": "#FFD700",
                **{name: AI_MODELS[name]["color"] for name in AI_MODELS},
            },
        )
        st.plotly_chart(fig3, use_container_width=True)

    # 4. 토론 요약 통계
    st.markdown("### 📈 토론 요약")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 메시지 수", sum(message_counts.values()))
    with col2:
        avg_msg_length = sum(
            d["total_chars"] for d in participation_data.values()
        ) / sum(message_counts.values())
        st.metric("평균 발언 길이", f"{avg_msg_length:.1f}자")
    with col3:
        debate_duration = (
            datetime.strptime(st.session_state.messages[-1]["time"], "%H:%M:%S")
            - datetime.strptime(st.session_state.messages[0]["time"], "%H:%M:%S")
        ).seconds / 60
        st.metric("토론 시간", f"{debate_duration:.1f}분")
    with col4:
        st.metric("참여자 수", len(participation_data))

        # 3초간 표시
    for i in range(100):
        # 진행률 업데이트
        progress_bar.progress(i + 1)
        status_text.write(f"🔍 분석 결과 표시 중... {i+1}%")
        time.sleep(0.15)  # 총 3초

    # 요소들 제거
    progress_bar.empty()
    status_text.empty()

    return participation_data


def main():
    """메인 애플리케이션"""
    # 페이지 기본 설정
    st.set_page_config(
        page_title="AI 토론 플랫폼",
        page_icon="💭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 스타일 적용
    apply_styles()

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ai_generator" not in st.session_state:
        st.session_state.ai_generator = AIResponseGenerator()
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "active" not in st.session_state:
        st.session_state.active = False
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None
    if "last_topic" not in st.session_state:
        st.session_state.last_topic = ""

    # 메인 타이틀 및 설명
    st.title("💭 AI 토론 플랫폼")
    st.markdown("#### GPT-4, Claude, Gemini와 함께하는 지적 대화")

    # 사이드바
    with st.sidebar:
        st.title("🗣️ 토론 참여자")
        for ai in AI_MODELS.values():
            st.markdown(
                f"""<div style='color:{ai["color"]}; 
                    margin:15px; padding:10px; 
                    background-color:{ai["color"]}10; 
                    border-radius:8px;'>
                    {ai["icon"]} <strong>{ai["name"]}</strong>
                    <div style='font-size:0.8em; margin-top:5px; 
                    color:#666;'>{ai["model"]}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        # 이전 세션 불러오기 옵션
        st.markdown("---")
        if not st.session_state.active:
            if load_session_from_json():
                st.success("이전 토론을 성공적으로 불러왔습니다!")

    # 분석 결과 표시 (토론 종료 후)
    if st.session_state.show_analysis:
        st.markdown(f"### 📊 토론 분석 결과: {st.session_state.last_topic}")
        analysis_data = analyze_debate_participation()

        # 새로운 토론 시작 버튼
        if st.button("🆕 새로운 토론 시작", use_container_width=True):
            st.session_state.active = False
            st.session_state.show_analysis = False
            st.session_state.messages = []
            st.session_state.topic = ""
            st.session_state.analysis_data = None
            st.rerun()
        return

    # 토론 시작 또는 진행
    if not st.session_state.active and not st.session_state.messages:
        with st.form(key="topic_form"):
            topic_input = st.text_input(
                "토론 주제를 입력하세요:",
                value="인공지능의 미래와 인류의 역할",
                key="topic_input",
            )
            if st.form_submit_button("🎯 토론 시작", use_container_width=True):
                st.session_state.topic = topic_input
                st.session_state.active = True
                st.rerun()

    # 토론 진행
    if st.session_state.active:
        st.markdown(f"### 📌 현재 주제: {st.session_state.topic}")
        display_messages()

        # 메시지 입력 및 제어 폼
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "의견을 입력하세요:", key="user_message", max_chars=500
            )

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                submit = st.form_submit_button("💬 전송", use_container_width=True)
            with col2:
                end_button = st.form_submit_button(
                    "⏹️ 토론 종료", use_container_width=True
                )
            with col3:
                save_button = st.form_submit_button(
                    "📥 토론 저장", use_container_width=True
                )

            if submit and user_input:
                process_user_input(user_input)
                st.rerun()

            if end_button:
                if len(st.session_state.messages) > 0:
                    # 토론 분석 실행
                    analysis_data = analyze_debate_participation()

                    # 세션 저장
                    filename = save_session_to_json()
                    if filename:
                        st.success(f"토론이 '{filename}'에 저장되었습니다.")

                st.session_state.active = False
                st.session_state.topic = ""
                st.session_state.messages = []
                st.rerun()

            if save_button:
                filename = save_session_to_json()
                if filename:
                    st.success(f"토론이 '{filename}'에 저장되었습니다.")

    else:
        st.info(
            "▶️ 토론 시작 버튼을 눌러 새로운 토론을 시작하거나, 이전 토론을 불러와주세요!"
        )


if __name__ == "__main__":
    main()


def show_api_status():
    """API 설정 상태를 확인하고 안내 메시지 표시"""
    if not st.session_state.ai_generator.initialization_successful:
        st.warning(
            """
        ⚠️ AI 응답 기능을 사용하기 위해서는 API 키 설정이 필요합니다.
        
        현재 사용 가능한 AI: {available_models}
        
        API 키 설정 방법:
        1. OpenAI API 키: https://platform.openai.com
        2. Anthropic API 키: https://console.anthropic.com
        3. Gemini API 키: https://makersuite.google.com/app/apikey
        
        설정된 API 키에 해당하는 AI만 토론에 참여할 수 있습니다.
        """.format(
                available_models=(
                    ", ".join(st.session_state.ai_generator.available_models)
                    if st.session_state.ai_generator.available_models
                    else "없음"
                )
            )
        )


class AIResponseGenerator:
    """AI 응답 생성을 관리하는 클래스"""

    def __init__(self):
        self.initialization_successful = False
        self.available_models = []

        try:
            # OpenAI API 초기화 시도
            if "OPENAI_KEY" in st.secrets:
                self.openai_client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
                self.available_models.append("지피")

            # Anthropic API 초기화 시도
            if "ANTHROPIC_KEY" in st.secrets:
                self.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_KEY"])
                self.available_models.append("로드")

            # Gemini API 초기화 시도
            if "GEMINI_KEY" in st.secrets:
                genai.configure(api_key=st.secrets["GEMINI_KEY"])
                self.available_models.append("재민")

            if self.available_models:
                self.initialization_successful = True
            else:
                st.warning("⚠️ API 키가 설정되지 않아 AI 응답 기능이 제한됩니다.")

        except Exception as e:
            st.error(f"API 초기화 오류: {str(e)}")
            self.initialization_successful = False

    def generate_response(self, ai_name, topic, messages):
        """AI 별 응답 생성"""
        if not self.initialization_successful:
            return "🔒 API 키가 설정되지 않아 응답할 수 없습니다."

        if ai_name not in self.available_models:
            return f"🔒 {ai_name}의 API 키가 설정되지 않아 응답할 수 없습니다."

        try:
            model_config = AI_MODELS[ai_name]
            prompt = self.format_prompt(messages, topic)

            # 각 AI 모델별 응답 생성 로직...

        except Exception as e:
            st.error(f"{ai_name} 응답 생성 오류: {str(e)}")
            return f"[시스템] {ai_name}의 응답 생성 중 오류가 발생했습니다."
