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
import requests

# AI 모델 설정 - DeepSeek 추가
AI_MODELS = {
    "지피": {
        "name": "지피",
        "icon": "🤖",
        "model": "gpt-4o-mini",
        "color": "#00A67E",
        "system_prompt": """당신은 논리적이고 분석적인 AI로서 적극적인 토론 참여자입니다.
        다른 참여자의 의견에 대해 건설적인 반론이나 보완점을 제시하세요.
        과학적 근거와 데이터를 바탕으로 대화에 참여하되,
        필요한 경우 다른 관점도 고려하여 균형 잡힌 시각을 보여주세요.""",
    },
    "로드": {
        "name": "로드",
        "icon": "🎩",
        "model": "claude-3-haiku-20240307",
        "color": "#7A5EA6",
        "system_prompt": """당신은 철학적이고 윤리적 관점을 중시하는 AI로서 열정적인 토론 참여자입니다.
        다른 참여자들의 의견에 대해 윤리적, 철학적 관점에서 심도 있는 분석을 제공하세요.
        특히 인간 가치와 도덕적 측면을 고려하여 토론을 더 깊이 있게 만드세요.""",
    },
    "재민": {
        "name": "재민",
        "icon": "🚀",
        "model": "gemini-2.0-flash",
        "color": "#4285F4",
        "system_prompt": """당신은 창의적이고 혁신적인 AI로서 활발한 토론 참여자입니다.
        기존의 틀을 벗어난 새로운 시각과 미래지향적 관점을 제시하세요.
        다른 참여자들의 의견을 바탕으로 더 발전된 아이디어를 제안하고,
        때로는 파격적인 제안을 통해 토론의 지평을 넓히세요.""",
    },
    "딥식": {
        "name": "딥식",
        "icon": "🧘",
        "model": "deepseek-chat",
        "color": "#E74C3C",
        "system_prompt": """당신은 동양철학에 정통하면서도 수학과 과학에 천재적인 통찰력을 지닌 AI입니다.
        논어와 도덕경의 지혜를 현대 과학기술과 접목시켜 새로운 관점을 제시하세요.
        복잡한 문제를 수학적 모델링과 동양철학의 균형잡힌 시각으로 분석하고,
        때로는 역설적 통찰을 통해 토론의 차원을 높이세요.""",
    },
}


class DeepSeekClient:
    """DeepSeek API 클라이언트"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate_response(self, prompt, system_message=""):
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return None
        except Exception as e:
            st.error(f"DeepSeek API 오류: {str(e)}")
            return None


class AIResponseGenerator:
    """AI 응답 생성을 관리하는 클래스"""

    def __init__(self):
        try:
            self.openai_client = OpenAI(api_key=st.secrets["OPENAI_KEY"])
            self.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_KEY"])
            self.deepseek_client = DeepSeekClient(st.secrets["DEEPSEEK_KEY"])
            genai.configure(api_key=st.secrets["GEMINI_KEY"])
            self.initialization_successful = True
        except Exception as e:
            st.error(f"API 초기화 오류: {str(e)}")
            self.initialization_successful = False

    def format_prompt(self, messages, topic, stance=None):
        """대화 내용을 프롬프트로 포맷팅"""
        recent_messages = messages[-3:]
        formatted_chat = "\n".join(
            [f"{msg['name']}: {msg['content']}" for msg in recent_messages]
        )

        stance_instruction = ""
        if stance:
            stance_instruction = f"\n당신의 입장: {stance}"

        prompt = f"""주제: {topic}{stance_instruction}

최근 대화:
{formatted_chat}

이 토론에 적극적으로 참여하여 다음을 수행해주세요:
1. 입장이 주어진 경우 해당 입장을 논리적으로 지지
2. 다른 참여자의 의견에 대한 분석적 피드백 제공
3. 새로운 관점이나 보완점 제시
4. 토론의 깊이를 더하는 질문이나 통찰 제공

응답은 300자 이내로 작성해주세요."""

        return prompt

    def generate_response(self, ai_name, topic, messages, stance=None):
        """각 AI 모델별 맞춤형 응답 생성"""
        try:
            model_config = AI_MODELS[ai_name]
            prompt = self.format_prompt(messages, topic, stance)

            if ai_name == "지피":
                response = self.openai_client.chat.completions.create(
                    model=model_config["model"],
                    messages=[
                        {"role": "system", "content": model_config["system_prompt"]},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                    temperature=0.8,
                )
                return response.choices[0].message.content[:300]

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
                return response.text[:300]

            elif ai_name == "딥식":
                response = self.deepseek_client.generate_response(
                    prompt, model_config["system_prompt"]
                )
                return response[:300] if response else None

        except Exception as e:
            st.error(f"{ai_name} 응답 생성 오류: {str(e)}")
            return f"[시스템] {ai_name}의 응답 생성 중 오류가 발생했습니다."


def create_message_html(msg):
    """메시지 HTML 생성"""
    is_user = msg["name"] == "사용자"
    if is_user:
        background_color = "#f1c40f20"
        icon = "👤"
    else:
        background_color = f"{AI_MODELS[msg['name']]['color']}20"
        icon = AI_MODELS[msg["name"]]["icon"]

    message_html = f"""
    <div class="message {'user' if is_user else 'ai'}">
        <div class="message-bubble" style="background-color: {background_color}">
            <div class="message-header">
                <strong>{msg['name']}</strong>
                <span class="ai-icon">{icon}</span>
                {f'<span class="stance-tag">{msg.get("stance", "")}</span>' if msg.get("stance") else ""}
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
    /* 기존 스타일 유지 */
    .stProgress > div > div > div {
        background-color: #4285f4;
        transition: width 0.3s ease;
    }
    
    .analysis-status {
        font-size: 1.2em;
        color: #4285f4;
        text-align: center;
        margin: 20px 0;
    }
    
    /* 채팅 컨테이너 스타일 업데이트 */
    .chat-container {
        max-width: 850px;
        margin: 20px auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
    }
    
    /* 메시지 스타일 업데이트 */
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
    
    /* 새로운 스타일: 입장 태그 */
    .stance-tag {
        font-size: 0.8em;
        padding: 2px 8px;
        border-radius: 10px;
        background-color: rgba(0,0,0,0.1);
        margin-left: 8px;
    }
    
    /* AI 아이콘 애니메이션 */
    .ai-icon {
        display: inline-block;
        animation: bounce 1s ease infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    
    /* 딥식 특별 스타일 */
    .ai[data-ai="딥식"] .message-bubble {
        border-left: 3px solid #E74C3C;
    }
    
    /* 반응형 디자인 개선 */
    @media (max-width: 768px) {
        .message-bubble {
            max-width: 90%;
        }
        
        .chat-container {
            margin: 10px;
            padding: 10px;
        }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def analyze_debate_participation():
    """토론 참여 분석 및 시각화"""
    if not st.session_state.messages:
        st.warning("분석할 토론 내용이 없습니다.")
        return

    # 진행률 표시줄 생성
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 분석 결과를 담을 데이터 구조
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

        # 상호작용 분석 (누구의 말에 응답했는지)
        if i > 0:
            prev_speaker = st.session_state.messages[i - 1]["name"]
            participation_data[name]["interactions"][prev_speaker] += 1

        # 진행률 업데이트
        progress = (i + 1) * 100 // len(st.session_state.messages)
        progress_bar.progress(progress)
        status_text.write(f"🔍 분석 진행 중... {progress}%")

    # 분석 결과 표시
    st.markdown(f"### 📊 토론 분석 결과: {st.session_state.topic}")

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

    # 분석 완료 표시
    progress_bar.empty()
    status_text.empty()
    st.success("✅ 토론 분석이 완료되었습니다!")

    return participation_data


def find_mentioned_ais(text):
    """텍스트에서 언급된 AI 찾기"""
    mentioned = []
    ai_names = ["지피", "로드", "재민", "딥식"]
    for name in ai_names:
        if name in text:
            mentioned.append(name)
    return mentioned


def process_user_input(user_input):
    """사용자 입력 처리 및 AI 응답 생성"""
    if not user_input.strip():
        return

    # 사용자 메시지 추가
    st.session_state.messages.append(
        {
            "name": "사용자",
            "content": user_input,
            "time": datetime.now().strftime("%H:%M:%S"),
            "icon": "👤",
        }
    )

    # 사용자가 언급한 AI 찾기
    mentioned_ais = find_mentioned_ais(user_input)

    # 언급된 AI가 있으면 해당 AI가 응답
    if mentioned_ais:
        for ai_name in mentioned_ais:
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

                # AI의 응답에서 다른 AI가 언급되었는지 확인
                mentioned_by_ai = find_mentioned_ais(response)
                if mentioned_by_ai:
                    # 이미 응답한 AI는 제외
                    mentioned_by_ai = [ai for ai in mentioned_by_ai if ai != ai_name]
                    if mentioned_by_ai:
                        next_ai = mentioned_by_ai[0]  # 첫 번째로 언급된 AI만 응답
                        follow_up = st.session_state.ai_generator.generate_response(
                            next_ai, st.session_state.topic, st.session_state.messages
                        )

                        if follow_up:
                            st.session_state.messages.append(
                                {
                                    "name": next_ai,
                                    "content": follow_up,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "icon": AI_MODELS[next_ai]["icon"],
                                }
                            )

    # 언급된 AI가 없으면 1-2개의 AI가 랜덤하게 응답
    else:
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

                # AI의 응답에서 다른 AI가 언급되었는지 확인
                mentioned_by_ai = find_mentioned_ais(response)
                if mentioned_by_ai:
                    # 이미 응답한 AI는 제외
                    mentioned_by_ai = [ai for ai in mentioned_by_ai if ai != ai_name]
                    if mentioned_by_ai:
                        next_ai = mentioned_by_ai[0]  # 첫 번째로 언급된 AI만 응답
                        follow_up = st.session_state.ai_generator.generate_response(
                            next_ai, st.session_state.topic, st.session_state.messages
                        )

                        if follow_up:
                            st.session_state.messages.append(
                                {
                                    "name": next_ai,
                                    "content": follow_up,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "icon": AI_MODELS[next_ai]["icon"],
                                }
                            )


def analyze_stance(text, for_stance, against_stance):
    """텍스트의 입장 분석"""
    # 간단한 키워드 기반 분석
    text = text.lower()
    for_keywords = for_stance.lower().split()
    against_keywords = against_stance.lower().split()

    for_score = sum(1 for word in for_keywords if word in text)
    against_score = sum(1 for word in against_keywords if word in text)

    return "찬성" if for_score > against_score else "반대"


def initialize_debate(topic, topic_type="일반", for_stance=None, against_stance=None):
    """토론 초기화"""
    st.session_state.topic = topic
    st.session_state.topic_type = topic_type
    st.session_state.messages = []
    st.session_state.active = True

    if topic_type == "찬반":
        st.session_state.for_stance = for_stance
        st.session_state.against_stance = against_stance

        # AI들의 입장 무작위 배정
        ais = list(AI_MODELS.keys())
        random.shuffle(ais)
        mid = len(ais) // 2

        st.session_state.ai_stances = {}
        for ai in ais[:mid]:
            st.session_state.ai_stances[ai] = "찬성"
        for ai in ais[mid:]:
            st.session_state.ai_stances[ai] = "반대"


def create_message_container(msg):
    """메시지 컨테이너 스타일 생성"""
    is_user = msg["name"] == "사용자"

    container = st.container()
    with container:
        if is_user:
            cols = st.columns([2, 10])
            with cols[1]:  # 오른쪽 정렬
                st.markdown(
                    f"""<div style='background-color: #f1c40f20; 
                    padding: 15px; border-radius: 15px; margin: 5px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong>{msg['name']}</strong>
                        <span>👤</span>
                    </div>
                    <div style='margin: 10px 0;'>{msg['content']}</div>
                    <div style='text-align: right; font-size: 0.8em; color: #666;'>
                        {msg['time']}
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            cols = st.columns([10, 2])
            with cols[0]:  # 왼쪽 정렬
                ai_color = AI_MODELS[msg["name"]]["color"]
                ai_icon = AI_MODELS[msg["name"]]["icon"]
                st.markdown(
                    f"""<div style='background-color: {ai_color}20; 
                    padding: 15px; border-radius: 15px; margin: 5px;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong>{msg['name']}</strong>
                        <span>{ai_icon}</span>
                    </div>
                    <div style='margin: 10px 0;'>{msg['content']}</div>
                    <div style='text-align: right; font-size: 0.8em; color: #666;'>
                        {msg['time']}
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def display_messages():
    """메시지 표시"""
    st.markdown(
        """
    <style>
        /* 메시지 컨테이너 스타일 */
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 15px;
        }
        /* 사용자/AI 아이콘 스타일 */
        .icon-container {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin: 5px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        create_message_container(msg)


def save_discussion():
    """토론 내용 저장"""
    if not st.session_state.messages:
        st.warning("저장할 토론 내용이 없습니다.")
        return

    try:
        # 저장할 데이터 준비
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_messages = []

        for msg in st.session_state.messages:
            clean_msg = {
                "name": msg["name"],
                "content": msg["content"],
                "time": msg["time"],
                "type": "user" if msg["name"] == "사용자" else "ai",
            }
            if msg["name"] != "사용자":
                clean_msg["ai_info"] = {
                    "model": AI_MODELS[msg["name"]]["model"],
                    "icon": AI_MODELS[msg["name"]]["icon"],
                }
            clean_messages.append(clean_msg)

        session_data = {
            "topic": st.session_state.topic,
            "timestamp": timestamp,
            "messages": clean_messages,
            "metadata": {
                "participants": list(
                    set(msg["name"] for msg in st.session_state.messages)
                ),
                "message_count": len(st.session_state.messages),
                "duration": str(
                    datetime.now()
                    - datetime.strptime(
                        st.session_state.messages[0]["time"], "%H:%M:%S"
                    )
                ),
            },
        }

        # JSON 문자열로 변환
        json_str = json.dumps(session_data, ensure_ascii=False, indent=2)

        # 파일명 생성
        filename = f"AI_토론_{timestamp}.json"

        # 다운로드 버튼 생성
        st.download_button(
            label="💾 토론 내용 다운로드",
            data=json_str.encode("utf-8"),
            file_name=filename,
            mime="application/json",
            key=f"download_{timestamp}",  # 유니크한 키 추가
        )

    except Exception as e:
        st.error(f"토론 저장 중 오류가 발생했습니다: {str(e)}")


def save_to_file(session_data, filename):
    """파일로 직접 저장"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"파일 저장 중 오류가 발생했습니다: {str(e)}")
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
                st.write(f"토론 시간: {session_data['metadata']['duration']}")

            return True

        except Exception as e:
            st.error(f"세션 불러오기 중 오류가 발생했습니다: {str(e)}")
            return False

    return False


def clear_session():
    """현재 세션 초기화"""
    if "messages" in st.session_state:
        st.session_state.messages = []
    if "topic" in st.session_state:
        st.session_state.topic = ""
    if "active" in st.session_state:
        st.session_state.active = False
    if "topic_type" in st.session_state:
        st.session_state.topic_type = "일반"
    if "for_stance" in st.session_state:
        st.session_state.for_stance = ""
    if "against_stance" in st.session_state:
        st.session_state.against_stance = ""
    if "ai_stances" in st.session_state:
        st.session_state.ai_stances = {}


def process_user_input(user_input):
    """사용자 입력 처리 및 AI 응답 생성"""
    if not user_input.strip():
        return

    # 사용자 메시지 추가
    st.session_state.messages.append(
        {
            "name": "사용자",
            "content": user_input,
            "time": datetime.now().strftime("%H:%M:%S"),
            "icon": "👤",
        }
    )

    # 1-2개의 AI가 랜덤하게 응답
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


def reset_session_state():
    """세션 상태 완전 초기화"""
    # 기존 AI 생성기는 유지
    ai_generator = (
        st.session_state.ai_generator if "ai_generator" in st.session_state else None
    )

    # 세션 상태 초기화
    st.session_state.clear()

    # AI 생성기 복원
    if ai_generator:
        st.session_state.ai_generator = ai_generator

    # 기본 상태 설정
    st.session_state.messages = []
    st.session_state.topic = ""
    st.session_state.active = False
    st.session_state.show_save = False
    st.session_state.show_analysis = False


def initialize_debate(topic):
    """토론 초기화"""
    st.session_state.topic = topic
    st.session_state.messages = []
    st.session_state.active = True
    st.session_state.show_save = False
    st.session_state.show_analysis = False


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
    if "show_save" not in st.session_state:
        st.session_state.show_save = False
    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    # 메인 타이틀 및 설명
    st.title("💭 AI 토론 플랫폼")
    st.markdown("#### GPT-4, Claude, Gemini, DeepSeek과 함께하는 지적 대화")

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

    # 재시작 버튼 (항상 표시)
    if st.button("🔄 새로운 토론 시작", use_container_width=True):
        reset_session_state()
        st.rerun()

    # 토론 시작 또는 진행
    if not st.session_state.active:
        with st.form(key="topic_form"):
            topic_input = st.text_input(
                "토론 주제를 입력하세요:", value="인공지능의 미래와 인류의 역할"
            )

            start_button = st.form_submit_button(
                "🎯 토론 시작", use_container_width=True
            )

            if start_button:
                initialize_debate(topic_input)
                st.rerun()

    # 토론 진행
    if st.session_state.active:
        st.markdown(f"### 📌 현재 주제: {st.session_state.topic}")
        display_messages()

        # 토론 저장 섹션 (폼 밖에 배치)
        if st.session_state.show_save:
            save_discussion()

        # 토론 분석 결과 표시
        if st.session_state.show_analysis:
            analyze_debate_participation()

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
                    st.session_state.show_analysis = True
                    st.session_state.show_save = True
                    st.rerun()

            if save_button:
                st.session_state.show_save = True
                st.rerun()


if __name__ == "__main__":
    main()
