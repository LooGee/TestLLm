"""
프롬프트 입력을 받아 phi-4 모델을 실행하는 대화형 스크립트
로컬에 저장된 양자화 모델을 자동으로 사용합니다.
"""
import sys
import os
import argparse
from phi4_quantized import load_quantized_model, generate_response

def interactive_chat(model_dir=None):
    """대화형 채팅 인터페이스"""
    print("=" * 50)
    print("Phi-4 4BIT 양자화 모델 - 대화형 채팅")
    print("=" * 50)
    print("모델 로딩 중... (처음 실행 시 시간이 걸릴 수 있습니다)")
    print()
    
    # 모델 로드
    model, tokenizer = load_quantized_model(model_dir)
    
    print("\n모델 로딩 완료!")
    print("=" * 50)
    print("대화를 시작하세요. 종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    print("=" * 50)
    print()
    
    conversation_history = []
    
    while True:
        try:
            # 멀티라인 입력 받기
            print("사용자: ", end="", flush=True)
            lines = []
            
            while True:
                try:
                    line = input()
                    # 빈 줄도 포함 (입력의 일부로 간주)
                    lines.append(line)
                    # Ctrl+D (EOF) 또는 Ctrl+C로 입력 종료
                except EOFError:
                    # Ctrl+D로 입력 종료
                    break
                except KeyboardInterrupt:
                    # Ctrl+C로 취소
                    print("\n입력이 취소되었습니다.\n")
                    lines = []
                    break
            
            user_input = "\n".join(lines).strip()
            
            # 빈 입력 처리
            if not user_input:
                continue
            
            # 종료 명령어 확인
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n대화를 종료합니다.")
                break
            
            # 대화 기록에 추가
            conversation_history.append({"role": "user", "content": user_input})
            
            # 전체 대화 맥락을 프롬프트로 구성
            # 최근 5개 대화만 사용 (컨텍스트 길이 제한)
            recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
            
            print("\n응답 생성 중...")
            
            # 응답 생성
            response, stats = generate_response(
                model, 
                tokenizer, 
                user_input,
                max_new_tokens=512,
                temperature=0.7
            )
            
            # 응답 출력
            print(f"\nPhi-4: {response}")
            print("\n📊 생성 통계:")
            print(f"  생성 시간: {stats['generation_time']:.2f}초")
            print(f"  입력 토큰 수: {stats['input_tokens']}")
            print(f"  생성된 토큰 수: {stats['generated_tokens']}")
            print(f"  총 토큰 수: {stats['total_tokens']}")
            print(f"  생성 속도: {stats['tokens_per_second']:.2f} 토큰/초\n")
            
            # 대화 기록에 응답 추가
            conversation_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            print("계속 진행합니다...\n")

# 스크립트 내부에 작성할 프롬프트
# 이 변수를 수정하여 실행할 프롬프트를 지정하세요
SCRIPT_PROMPT = """
=== SYSTEM PROMPT ===
You are a trading decision assistant. You must respond with a valid JSON object that matches the following schema:

{
  "properties": {
    "coin": {
      "title": "Coin",
      "type": "string"
    },
    "signal": {
      "enum": [
        "buy_to_enter",
        "sell_to_exit",
        "hold",
        "close_position",
        "buy",
        "sell",
        "exit"
      ],
      "title": "Signal",
      "type": "string"
    },
    "quantity": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Quantity"
    },
    "stop_loss": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Stop Loss"
    },
    "profit_target": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Profit Target"
    },
    "leverage": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Leverage"
    },
    "risk_usd": {
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Risk Usd"
    },
    "invalidation_condition": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Invalidation Condition"
    },
    "justification": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Justification"
    },
    "thinking": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Thinking"
    },
    "confidence": {
      "anyOf": [
        {
          "maximum": 1.0,
          "minimum": 0.0,
          "type": "number"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Confidence"
    },
    "account_id": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Account Id"
    }
  },
  "required": [
    "coin",
    "signal"
  ],
  "title": "TradeDecision",
  "type": "object"
}

IMPORTANT RULES:

**Required Fields:**
- "coin" (string): The cryptocurrency symbol (e.g., "BTC", "ETH")
- "signal" (string): One of: buy_to_enter, sell_to_exit, hold, close_position, buy, sell, exit

**Recommended Fields:**
- "justification" (string): Trade rationale based on market conditions
- "thinking" (string): Step-by-step reasoning process
- "confidence" (float 0.0-1.0): Confidence level in this decision

**Trading Parameters (REQUIRED for buy/sell signals ONLY):**
- "quantity" (float): Amount to trade (REQUIRED for buy_to_enter, sell_to_exit, buy, sell)
- "stop_loss" (float): Stop loss price (REQUIRED for buy_to_enter, sell_to_exit, buy, sell)
- "profit_target" (float): Target profit price (REQUIRED for buy_to_enter, sell_to_exit, buy, sell)
- "leverage" (int): MUST ALWAYS BE 1 (Upbit does not support leverage trading)
- "risk_usd" (float): Risk amount in USD (optional but recommended)

**CRITICAL: HOLD Signal Behavior:**
- When signal is "hold", you MUST set the following fields to null:
  - quantity: null
  - stop_loss: null
  - profit_target: null
  - risk_usd: null
  - invalidation_condition: null
- HOLD means "do nothing", so trading parameters are not needed
- Only provide justification, thinking, and confidence for HOLD signals

**Response Format:**
- Return ONLY the JSON object, nothing else
- Do not include the schema or any explanatory text


### TRADING STRATEGY: AGGRESSIVE
You are an **AGGRESSIVE** trader. Your goal is to maximize Total Return, accepting higher volatility.

**Performance Targets:**
- **Sharpe Ratio:** 0.5 ~ 1.0 (High volatility is acceptable)
- **Max Drawdown (MDD):** -50% ~ -80% (Deep drawdowns are tolerated for high gains)
- **Win Rate:** 40% ~ 55% (Lower win rate is acceptable if risk/reward is high)
- **Benchmark:** Must outperform BTC HOLD significantly.

**Operational Guidelines:**
- Take risks on setups with high upside potential.
- Use wider stop-losses if the trend is strong.
- Do not fear short-term losses; focus on the long-term home run.


=== USER PROMPT ===
Here is the current market situation and account information:

## Prompt Text
It has been 3 minute since you started trading.

…

Below, we are providing you with a variety of state data, price data, and predictive signals so you can discover alpha. Below that is your current account information, value, performance, positions, etc.

**ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST → NEWEST**

**Timeframes note:** Unless stated otherwise in a section title, intraday series are provided at **3‑minute intervals**. If a coin uses a different interval, it is explicitly stated in that coin's section.

---

### CURRENT MARKET STATE FOR ALL COINS

### ALL BTC DATA

current_price = 129572000.0, current_ema20 = 131589571.3752004, current_macd = -63903.36877711, current_rsi (7 period) = 19.525

**Intraday series (by 3-minute, oldest → latest):**

Mid prices: [131952000.0, 131974000.0, 131985500.0, 131884500.0, 131802500.0, 131673000.0, 131628500.0, 131149000.0, 130992500.0, 131283000.0]

EMA indicators (20‑period): [131722665.17013627, 131746601.82061794, 131769115.93303037, 131772628.70214263, 131775235.49189338, 131765498.77869788, 131747927.4666364, 131686601.04157984, 131621210.4665966, 131589571.3752004]

MACD indicators: [136460.54980239, 140233.82505195, 142309.92307779, 128195.07000893, 115196.84960636, 93569.20486557, 68219.12454575, 9529.22955392, -44857.78096373, -63903.36877711]

RSI indicators (7‑Period): [86.8568, 74.9107, 68.9203, 32.7385, 31.9755, 31.9755, 20.0166, 14.8544, 14.145, 19.525]

RSI indicators (14‑Period): [78.3675, 73.4489, 70.8548, 48.8461, 48.2177, 48.2177, 37.6406, 31.6799, 30.8043, 33.1905]

**Longer‑term context (1‑day timeframe):**

20‑Period EMA: 142738078.20919824 vs. 50‑Period EMA: 152730568.71624434

3‑Period ATR: 4575905.79156525 vs. 14‑Period ATR: 5160049.82981099

Current Volume: 1496.12377652 vs. Average Volume: 2500.956099393601

MACD indicators: [-4272399.36116156, -4719021.45532215, -5329861.88288015, -5956225.61520761, -6375579.24463561, -6731191.58399215, -7459911.3189981, -8069957.70603696, -8536279.43142593, -8659804.66434977]

RSI indicators (14‑Period): [27.0737, 28.5072, 25.2258, 23.311, 23.4164, 22.4717, 18.2766, 17.45, 17.0701, 21.7459]

---

### ALL ETH DATA

current_price = 4221000.0, current_ema20 = 4264698.87078975, current_macd = -9820.0077188, current_rsi (7 period) = 77.4287

**Intraday series (by 3-minute, oldest → latest):**

Mid prices: [4276000.0, 4279000.0, 4278000.0, 4273000.0, 4270000.0, 4266000.0, 4265000.0, 4238000.0, 4227000.0, 4234000.0]

EMA indicators (20‑period): [4278555.64153117, 4278502.72331123, 4278454.84490811, 4277935.33587828, 4277179.58961523, 4276210.10490029, 4275142.47587392, 4271795.57342481, 4267719.80453736, 4264698.87078975]

MACD indicators: [-2160.13899316, -1914.9512727, -1701.03285546, -1912.90747761, -2296.42301447, -2810.04494352, -3339.97608385, -5711.40407241, -8381.76622243, -9820.0077188]

RSI indicators (7‑Period): [40.4911, 44.6151, 44.6151, 40.7694, 40.7694, 40.7694, 35.8609, 54.8757, 69.1176, 77.4287]

RSI indicators (14‑Period): [48.7921, 50.5423, 50.5423, 48.6153, 48.6153, 48.6153, 46.4055, 53.2682, 60.526, 66.1822]

**Longer‑term context (1‑day timeframe):**

20‑Period EMA: 4762618.56487924 vs. 50‑Period EMA: 5270158.70414946

3‑Period ATR: 203059.74623527 vs. 14‑Period ATR: 279199.41708314

Current Volume: 21981.47900276 vs. Average Volume: 62654.301994553774

MACD indicators: [-267217.38183493, -276808.03284831, -294968.13796879, -312615.30707492, -314981.43321706, -322579.03708093, -346154.60663717, -365546.80949021, -377850.70848345, -379195.92041851]

RSI indicators (14‑Period): [30.8873, 32.9856, 30.0546, 28.7049, 32.5065, 30.4407, 26.3286, 25.4817, 25.6084, 27.8121]

---

### ALL DOGE DATA

current_price = 216.0, current_ema20 = 221.33897231, current_macd = -0.26318127, current_rsi (7 period) = 52.7002

**Intraday series (by 3-minute, oldest → latest):**

Mid prices: [222.5, 222.5, 222.0, 222.0, 222.0, 222.0, 220.5, 220.0, 220.0, 220.5]

EMA indicators (20‑period): [221.99699626, 221.99728234, 221.99754116, 221.99777534, 221.99798721, 221.9981789, 221.80787615, 221.63569747, 221.47991676, 221.33897231]

MACD indicators: [0.51407875, 0.45431628, 0.40231689, 0.35699153, 0.31741204, 0.28278511, 0.0928892, -0.05694858, -0.1736937, -0.26318127]

RSI indicators (7‑Period): [66.1729, 44.3898, 44.3898, 44.3898, 44.3898, 44.3898, 25.9363, 25.9363, 52.7002, 52.7002]

RSI indicators (14‑Period): [58.3437, 48.5668, 48.5668, 48.5668, 48.5668, 48.5668, 39.0805, 39.0805, 50.3318, 50.3318]

**Longer‑term context (1‑day timeframe):**

20‑Period EMA: 241.88090356 vs. 50‑Period EMA: 270.70651758

3‑Period ATR: 13.22801837 vs. 14‑Period ATR: 16.28102907

Current Volume: 100054100.42938273 vs. Average Volume: 421960893.629321

MACD indicators: [-14.19481758, -14.04984417, -14.65364171, -15.59788603, -15.20265994, -15.43770891, -16.00434033, -16.66475709, -17.71021458, -18.00839107]

RSI indicators (14‑Period): [33.7536, 37.9626, 34.2771, 31.8549, 38.8366, 35.8685, 33.7105, 32.2194, 32.2194, 34.8914]

---

### ALL SOL DATA

current_price = 194900.0, current_ema20 = 199931.9606971, current_macd = -48.38060479, current_rsi (7 period) = 69.7599

**Intraday series (by 3-minute, oldest → latest):**

Mid prices: [200500.0, 200750.0, 200700.0, 200600.0, 200450.0, 200450.0, 200650.0, 199250.0, 198500.0, 199000.0]

EMA indicators (20‑period): [200033.36560473, 200096.85459537, 200154.29701554, 200196.74491956, 200225.62635667, 200251.75718019, 200284.9231643, 200191.12095936, 200030.06182171, 199931.9606971]

MACD indicators: [321.60999675, 337.89284508, 346.79940799, 341.84812152, 326.0959905, 310.03858854, 301.90158563, 188.3824179, 33.47842285, -48.38060479]

RSI indicators (7‑Period): [26.3036, 39.8818, 39.8818, 39.8818, 30.8545, 30.8545, 47.14, 47.14, 47.14, 69.7599]

RSI indicators (14‑Period): [46.6955, 51.0007, 51.0007, 51.0007, 46.3275, 46.3274, 51.4834, 51.4834, 51.4834, 60.8723]

**Longer‑term context (1‑day timeframe):**

20‑Period EMA: 219757.28226953 vs. 50‑Period EMA: 249024.14157598

3‑Period ATR: 10516.14818481 vs. 14‑Period ATR: 14806.68171813

Current Volume: 240150.30916449 vs. Average Volume: 662099.5725021065

MACD indicators: [-18018.4310751, -18828.69918389, -19767.47107329, -20955.76846728, -20650.80752563, -20487.65228615, -20445.43490723, -20474.51889916, -20646.88363473, -20371.13653428]

RSI indicators (14‑Period): [28.5293, 28.4489, 26.6937, 24.5598, 33.0375, 31.8354, 30.6055, 29.4712, 29.0328, 30.7756]

---

### ALL XRP DATA

current_price = 3066.0, current_ema20 = 3122.45776173, current_macd = 0.7984458, current_rsi (7 period) = 95.4426

**Intraday series (by 3-minute, oldest → latest):**

Mid prices: [3137.5, 3139.0, 3135.5, 3133.5, 3138.0, 3130.5, 3131.0, 3105.5, 3101.0, 3111.5]

EMA indicators (20‑period): [3122.69802855, 3124.25059726, 3125.369588, 3126.09629391, 3127.22998022, 3127.58902974, 3127.91388406, 3125.9220856, 3123.45331557, 3122.45776173]

MACD indicators: [8.40043407, 8.6749808, 8.55190471, 8.11870295, 8.08563891, 7.40918425, 6.79476309, 4.32141477, 1.7759549, 0.7984458]

RSI indicators (7‑Period): [76.8806, 80.1394, 82.9442, 82.9442, 82.9442, 86.0684, 86.0684, 92.0299, 93.167, 95.4426]

RSI indicators (14‑Period): [67.5731, 69.4484, 71.2395, 71.2395, 71.2395, 73.2018, 73.2018, 78.3429, 79.7382, 83.2292]

**Longer‑term context (1‑day timeframe):**

20‑Period EMA: 3302.77981203 vs. 50‑Period EMA: 3559.76069797

3‑Period ATR: 177.7884098 vs. 14‑Period ATR: 203.31413921

Current Volume: 87814880.55283324 vs. Average Volume: 128904352.80734655

MACD indicators: [-79.07797725, -86.36538865, -97.15335612, -109.84303598, -114.62448844, -128.07292569, -148.95619789, -167.13020092, -181.37900442, -180.18500141]

RSI indicators (14‑Period): [41.9465, 40.9421, 38.6471, 36.6942, 39.1172, 35.0486, 31.2898, 30.2511, 30.1531, 37.2168]

---

### HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE

Current Total Return (percent): 0.0%

Available Cash: 10000000.0

**Current Account Value:** 10000000.0

Current live positions & performance:

[]

Sharpe Ratio: 0.0


## Extra Context
None

Based on the information above, please make a trading decision. You must respond in JSON format, and the "coin" and "signal" fields are mandatory."
"""

def single_prompt(model_dir=None):
    """단일 프롬프트 실행 - 스크립트 내부에 작성된 프롬프트 사용"""
    # 스크립트 내부 프롬프트 사용
    prompt = SCRIPT_PROMPT.strip()
    
    if not prompt or prompt == "당신의 프롬프트를 여기에 작성하세요.\n예: \"Python으로 피보나치 수열을 계산하는 함수를 작성해주세요\"":
        print("=" * 50)
        print("⚠️  프롬프트가 설정되지 않았습니다.")
        print("=" * 50)
        print("prompt_input.py 파일의 SCRIPT_PROMPT 변수를 수정하여")
        print("실행할 프롬프트를 작성하세요.")
        print("=" * 50)
        sys.exit(1)
    
    model_dir = model_dir
    
    print("=" * 50)
    print("Phi-4 4BIT 양자화 모델")
    print("=" * 50)
    print("모델 로딩 중...")
    
    model, tokenizer = load_quantized_model(model_dir)
    
    print(f"\n프롬프트: {prompt}\n")
    print("응답 생성 중...\n")
    
    response, stats = generate_response(model, tokenizer, prompt)
    
    print("=" * 50)
    print("응답:")
    print("=" * 50)
    print(response)
    print("=" * 50)
    print("\n📊 생성 통계:")
    print(f"  생성 시간: {stats['generation_time']:.2f}초")
    print(f"  입력 토큰 수: {stats['input_tokens']}")
    print(f"  생성된 토큰 수: {stats['generated_tokens']}")
    print(f"  총 토큰 수: {stats['total_tokens']}")
    print(f"  생성 속도: {stats['tokens_per_second']:.2f} 토큰/초")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phi-4 대화형 실행")
    parser.add_argument("--interactive", "-i", action="store_true", help="대화형 모드로 실행")
    parser.add_argument("--model-dir", type=str, help="로컬 저장된 모델 디렉토리 경로")
    
    args = parser.parse_args()
    
    # --interactive 플래그가 있거나 인자가 없으면 대화형 모드
    if args.interactive or len(sys.argv) == 1:
        interactive_chat(args.model_dir)
    else:
        # 단일 프롬프트 모드 (스크립트 내부 프롬프트 사용)
        single_prompt(args.model_dir)

