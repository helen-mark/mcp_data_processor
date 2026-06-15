import atexit
from enum import Enum
import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Optional, Dict, Any

import ollama
import openai
import yaml

import pandas as pd

class DataType(Enum):
    MAIL = "mail"
    CALLS = "calls"

class CsvProcessor:
    def __init__(self, model: str, output_csv_path: str, batch_size: int = 100, data_type: DataType = DataType.MAIL, config_path: str = 'config.yml'):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        self.model = model  #'opt/models/openai-gpt-oss-120b/'
        self.vllm_port = 8000  # CURRENTLY UNUSED
        self.vllm_process = None  # CURRENTLY UNUSED
        self.startup_timeout = 1000
        #self._start_vllm_server()

        self.output_csv_path = output_csv_path
        self.batch_size = batch_size
        self.tags_list = config.get('tags_list', [])
        self.data_type = data_type
        self.is_local = False
        self.client = ollama.Client(headers={"Connection": "keep-alive"})
        #self.client = openai.OpenAI(
        #    base_url="http://localhost:8000/vllm-server",  # vLLM server endpoint
        #    api_key="EMPTY",  # vLLM doesn't require API key
        #    timeout=120.0,
        #    max_retries=3,
        #)
        self.processed_files = set()

    # CURRENTLY UNUSED
    def _detect_dtype(self):
        config_path = os.path.join(self.model, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    quantization = config.get("quantization_config", {})
                    if quantization:
                        quant_method = quantization.get("quant_method", "")
                        print(f"Detected quantization: {quant_method}")
                        if "mxfp4" in quant_method.lower():
                            return "bfloat16"
                        elif "awq" in quant_method.lower():
                            return "float16"
                        elif "gptq" in quant_method.lower():
                            return "float16"
            except Exception as e:
                print(f"Warning: Could not detect quantization: {e}")
        return None  # Let vLLM auto-detect

    # CURRENTLY UNUSED
    def _detect_quantization(self):
        config_path = os.path.join(self.model, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                quantization = config.get("quantization_config", {})
                if quantization:
                    quant_method = quantization.get("quant_method", "")
                    print(f"Detected quantization: {quant_method}")
                    if "mxfp4" in quant_method:
                        return "bfloat16"
                    elif "awq" in quant_method:
                        return "float16"
                    elif "gptq" in quant_method:
                        return "float16"
        return "bfloat16"  # Default to bfloat16 for safety

    # CURRENTLY UNUSED
    def _start_vllm_server(self, gpu_memory_util: float = 0.9, max_model_len: int = 4096):
        print(f"\n{'='*60}")
        print(f"Starting vLLM server")
        print(f"Model: {self.model}")
        print(f"Port: {self.vllm_port}")
        print(f"GPU Memory Utilization: {gpu_memory_util}")
        print(f"Max Model Length: {max_model_len}")
        print(f"Startup Timeout: {self.startup_timeout} seconds")
        print(f"{'='*60}\n")

        if not os.path.exists(self.model):
            raise FileNotFoundError(f"Model not found: {self.model}")

        dtype = self._detect_dtype()
        if dtype:
            print(f"Using dtype: {dtype}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.vllm_port),
            "--gpu-memory-utilization", str(gpu_memory_util),
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", "256",
            "--dtype", "bfloat16",  # GPT-OSS требует bfloat16 для MXFP4 [citation:2]
            "--trust-remote-code",  # Add trust-remote-code for some models
            "--disable-log-requests",
            "--disable-log-stats",
        ]

        env = os.environ.copy()
        #env.update({
        #"VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        #"VLLM_USE_V1": "1",
        #"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        #"TORCHINDUCTOR_COMPILE_THREADS": "4",
        #"VLLM_LOGGING_LEVEL": "WARNING",
        #})

        env.update({
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "CUDA_VISIBLE_DEVICES": "0",  # Явно указываем GPU
        "VLLM_LOGGING_LEVEL": "INFO",
        "VLLM_USE_TRITON": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # Поддержка разных GPU
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "PATH": f"/usr/local/cuda-12.8/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"/usr/local/cuda-12.8/lib64:{env.get('LD_LIBRARY_PATH', '')}",
        })

        if dtype:
            cmd.extend(["--dtype", dtype])

        #if self.verbose:
        #    print("Command:", " ".join(cmd))
        #    print()

        try:
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()
            )

            self._start_log_capture()

            atexit.register(self._cleanup_server)

            # Wait for server to be ready
            if not self._wait_for_server():
                raise RuntimeError("vLLM server failed to start within timeout")

        except Exception as e:
            print(f"❌ Failed to start vLLM server: {e}")
            self._cleanup_server()
            raise

    def _start_log_capture(self):
        def capture_logs():
            if not self.vllm_process:
                return
            
            try:
                # Read stderr line by line
                for line in iter(self.vllm_process.stderr.readline, ''):
                    if not line:
                        break
                    
                    line = line.strip()
                    if line:
                        # Format and print logs with timestamp
                        timestamp = time.strftime("%H:%M:%S")
                    
                        # Color-code different log levels
                        if "ERROR" in line or "error" in line:
                            print(f"\033[91m[{timestamp}] [vLLM ERROR] {line}\033[0m")
                        elif "WARNING" in line or "Warning" in line:
                            print(f"\033[93m[{timestamp}] [vLLM WARNING] {line}\033[0m")
                        elif "INFO" in line:
                            # Filter out less important info messages
                            if not any(skip in line for skip in ["DEBUG", "heartbeat"]):
                                print(f"\033[92m[{timestamp}] [vLLM INFO] {line}\033[0m")
                        else:
                            print(f"[{timestamp}] [vLLM] {line}")
                        
            except Exception as e:
                print(f"Error capturing logs: {e}")
            finally:
                # Close stderr when done
                if self.vllm_process and self.vllm_process.stderr:
                    self.vllm_process.stderr.close()
    
        # Start daemon thread so it exits when main process exits
        self.log_thread = threading.Thread(target=capture_logs, daemon=True)
        self.log_thread.start()

    # CURRENTLY UNUSED
    def _wait_for_server(self, max_retries=400, retry_delay=2):
        print("Waiting for vLLM server to start...")

        for i in range(max_retries):
            try:
                import requests
                response = requests.get(f"http://localhost:{self.vllm_port}/health", timeout=5)
                if response.status_code == 200:
                    print(f"vLLM server is ready after {i+1} seconds")
                    return True
            except:
                pass

            if self.vllm_process and self.vllm_process.poll() is not None:
                stderr = self.vllm_process.stderr.read()
                print(f"vLLM server died with error: {stderr}")
                return False

            time.sleep(retry_delay)
            print(f"   Waiting... ({i+1}/{max_retries})")

        print("vLLM server failed to start within timeout")
        return False

    # CURRENTLY UNUSED
    def _cleanup_server(self):
        if self.vllm_process:
            print("\n Shutting down vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
            print("vLLM server stopped")

    #def __del__(self):
    #    """Destructor to ensure cleanup"""
    #    self._cleanup_server()

    def _run_tagging_locally(self, text: str) -> Dict[str, Any]:
        try:
            return self.get_tags_from_llm(text)
        except Exception as e:
            print(f"Ошибка тегирования: {e}")
            return {'result': [], 'summary': f'Ошибка: {str(e)}'}

    def process(self, add_tags: bool = False, input_csv_path: Optional[str] = None) -> None:
        csv_path = input_csv_path if input_csv_path else self.output_csv_path

        df = pd.read_csv(csv_path)

        if 'tags' not in df.columns:
            df['tags'] = None

        mask_empty = ~df['text'].isna() & (df['tags'].isna() | (df['tags'] == ''))
        indices_to_process = df.index.tolist() if add_tags else df[mask_empty].index.tolist()

        if not indices_to_process:
            print("Все строки уже имеют теги.")
            return

        print(f"Найдено {len(indices_to_process)} строк без тегов.")

        for i in range(0, len(indices_to_process), self.batch_size):
            batch_indices = indices_to_process[i:i + self.batch_size]
            batch_df = df.loc[batch_indices]

            print(f"Обработка батча {i // self.batch_size + 1}, размер: {len(batch_indices)}")

            for idx in batch_indices:
                text = df.loc[idx, 'text']
                if pd.isna(text) or text == '':
                    df.at[idx, 'tags'] = '[]'
                    df.at[idx, 'summary'] = 'нет'
                    continue

                print(text[:200])

                try:
                    result = self.get_single_tag_from_llm(text) if add_tags else self._run_tagging_locally(text)

                    if 'result' in result:
                        if add_tags:
                            tags_str = df.at[idx, 'tags'].replace("'", '"')
                            tags = result['result'] + json.loads(tags_str)
                        else:
                            tags = result['result']
                        df.at[idx, 'tags'] = str(tags)
                    elif not add_tags:
                        df.at[idx, 'tags'] = '[]'
                    if 'summary' in result:
                        s = result['summary']
                        df.at[idx, 'summary'] = str(s)
                    else:
                        df.at[idx, 'summary'] = 'нет'


                except Exception as e:
                    print(f"Ошибка обработки индекса {idx}: {e}")
                    if not add_tags:
                        df.at[idx, 'tags'] = '[]'
                    df.at[idx, 'summary'] = 'нет'

                time.sleep(0.1)

            df.to_csv(self.output_csv_path, index=False)
            print(f"Батч {i // self.batch_size + 1} обработан и сохранен.")
            print(df.columns)
            print(len(df))

        print("Обработка завершена.")

    def get_single_tag_from_llm(self, text: str) -> Dict[str, Any]:
            truncated_text = text[:3000] + "..." if len(text) > 3000 else text

            prompt = f"""Ты — специалист по категоризации транскрибированных телефонных разговоров и писем электронной почты.
    Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

    Вот текст одного разговора или письма:
    {truncated_text}

    ТВОЕ ЗАДАНИЕ: определи, есть ли в этом тексте жалоба клиента на конкретного менеджера, а именно: либо упоминание фамилии или имени менеджера и выражение недовольства,
    либо упоминание конкретного менеджера без названия его имени, например: "наш менеджер совершенно некомпетентен..." или: Этот менеджер, с которым мы регулярно общаемся,
    постоянно не отвечает на письма..." или: "тот оператор, который занимается нами, в прошлый раз мне нагрубил...". То есть ты ищешь не просто описание безликой проблемы,
    а прямое указание на исполнителя.

    ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
    {{
      "result": ["жалоба на менеджера"]
    }}
    
    ЛИБО, если прямого указания на менеджера с описанием проблемы не найдено в тексте, то список должен быть пустым:
    {{
      "result": []
    }}
    """

            empty_response = {
                "result": [],
                "summary": '',
                "additional_tags": [],
                "reasoning": "Ошибка"
            }

            try:
                if len(truncated_text) < 30:
                    print('Text is too short')
                    return empty_response
                # if self.is_local:
                #     response = self.model(prompt_mail if self.mail else prompt,
                #                           temperature=0.3,
                #                           top_p=0.9,
                #                           # num_gpu=-1,
                #                           num_ctx=4096)
                # else:
                print('Getting response ...')
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    keep_alive=-1,
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                        # "num_gpu": -1,
                        'num_ctx': 4096
                    }
                )
                # response = self.client.chat.completions.create(
                # model=self.model,
                # messages=[
                # {"role": "user", "content": prompt_mail if self.mail else prompt}
                # ],
                # temperature=0.3,
                # top_p=0.9,
                # max_tokens=512,
                # )

                # response_text = response.choices[0].message.content

                response_text = response['response']
                print('response + ctx8: ', response_text)
                # token_count = response['prompt_eval_count']
                # print(f"Real n tokens in prompt: {token_count}")

                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    valid_selected = []
                    for tag in result.get('result', []):
                        if tag in self.tags_list:
                            valid_selected.append(tag)
                        else:
                            print(f"Модель придумала тег '{tag}', игнорирую")
                    valid_selected.append(self.data_type.value)
                    print(valid_selected)

                    result['result'] = valid_selected
                    return result
                else:
                    raise ValueError("LLM не вернула JSON")

            except Exception as e:
                print(f"Ошибка при запросе к LLM: {e}")
                return empty_response

    def get_tags_from_llm(self, text: str) -> Dict[str, Any]:
        truncated_text = text[:3000] + "..." if len(text) > 3000 else text

        prompt = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Во-первых, Проанализируй этот разговорный текст и верни краткое summary, характеризующие 1-2 главных причины обращения клиента.

Во-вторых, Проанализируй этот разговорный текст. Ознакомься со списком заранее сформированных описаний: 
{', '.join(self.tags_list)}. Подходят ли какие-нибудь из них этому тексту?
Присвой тексту от 0 до 3 описаний из фиксированного списка, только если они действительно хорошо характеризуют причины обращения клиента.
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда присвой два описания: про долгое ожидание ответа и про недоставку (несвоевременную замену).
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда подойдет описание про возобновление услуг и описание про добавление ковров. И так далее.

Если клиент выражает недовольство ценами или непонимание, почему цены неожиданно выросли, - выбирай описание "клиент недоволен ценами". Но если клиент просто запрашивает информацию о планируемом росте цен, или уточняет, когда будет индексация, не выражая непонимания или недовольства, - то выбирай описание, связанное с уточнением деталей. Сам факт роста цен в связи с инфляцией - нормален.
Аналогично с другими проблемами: если клиент упоминает ключевые слова, связанные с какой-либо проблемой, - это не всегда означает факт возникновения проблемы. Будь внимателен!

Выбирай описание "консультация или уточнение деталей" только в случае, если нет никакой другой причины обращения!
Если ни одно описание не подходит, - просто не присваивай никаких описаний.

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["описание1", "описание2"],
  "summary": "причины обращения клиента своими словами"
}}
Если текст не содержит ясной причины обращения - верни пустой список описаний и слово "нет" в качестве summary.
"""

        prompt_mail = f"""Ты — специалист по категоризации писем электронной почты.
Есть емейл сообщения от клиентов, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот один текст:
{truncated_text}

ТВОЕ ЗАДАНИЕ: Во-первых, Проанализируй этот разговорный текст и верни краткое summary, характеризующие 1-2 главных причины обращения клиента.

Во-вторых, Проанализируй этот текст. Ознакомься со списком заранее сформированных описаний: 
{', '.join(self.tags_list)}. Подходят ли какие-нибудь из них этому тексту?
Присвой тексту от 0 до 3 описаний из фиксированного списка, только если они действительно хорошо характеризуют причины обращения клиента. 
Например, клиент долго не получает ответ на его заявку о том, что ему не доставили вовремя ковер. Тогда присвой два описания: "долго нет ответа на заявку" и "не заменили ковры вовремя".
Либо клиент хочет возобновить услуги И при этом добавить больше ковров, чем было у него раньше. Тогда подойдет описание про возобновление услуг и описание про добавление ковров. И так далее.

Если клиент выражает недовольство ценами или непонимание, почему цены неожиданно выросли, - выбирай описание "клиент недоволен ценами". Но если клиент просто запрашивает информацию о планируемом росте цен, или уточняет, когда будет индексация, не выражая непонимания или недовольства, - то выбирай описание, связанное с уточнением деталей. Сам факт роста цен в связи с инфляцией - нормален.
Аналогично с другими проблемами: если клиент упоминает ключевые слова, связанные с какой-либо проблемой, - это не всегда означает факт возникновения проблемы. Будь внимателен!

Выбирай описание "консультация или уточнение деталей" только в случае, если нет никакой другой причины обращения!
Если ни одно описание не подходит, - просто не присваивай никаких описаний. Не придумывай описания! Используй только данный тебе список.

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON:
{{
  "result": ["описание1", "описание2"],
  "summary": "причины обращения клиента своими словами"
}}
Если текст не содержит ясной причины обращения - верни пустой список описаний и слово "нет" в качестве summary.
"""
    # print(prompt_mail)
    #         import psutil
    #         import GPUtil

    #         def check_resources():
    #             # CPU и RAM
    #             print(f"CPU: {psutil.cpu_percent()}%")
    #             print(f"RAM: {psutil.virtual_memory().percent}%")

    #             # GPU
    #             try:
    #                 gpus = GPUtil.getGPUs()
    #                 for gpu in gpus:
    #                     print(f"GPU {gpu.name}: {gpu.memoryUtil*100:.1f}% used")
    #             except:
    #                 pass
    #         check_resources()

        empty_response = {
            "result": [],
            "summary": '',
            "additional_tags": [],
            "reasoning": "Ошибка"
        }

        try:
            if len(truncated_text) < 30:
                print('Text is too short')
                return empty_response
            # if self.is_local:
            #     response = self.model(prompt_mail if self.mail else prompt,
            #                           temperature=0.3,
            #                           top_p=0.9,
            #                           # num_gpu=-1,
            #                           num_ctx=4096)
            # else:
            print('Getting response ...')
            response = self.client.generate(
                model=self.model,
                prompt=prompt_mail if self.data_type == DataType.MAIL else prompt,
                keep_alive=-1,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    # "num_gpu": -1,
                    'num_ctx': 4096
                }
            )
            #response = self.client.chat.completions.create(
            #model=self.model,
            #messages=[
            #{"role": "user", "content": prompt_mail if self.mail else prompt}
            #],
            #temperature=0.3,
            #top_p=0.9,
            #max_tokens=512,
            #)

            #response_text = response.choices[0].message.content

            response_text = response['response']
            print('response + ctx8: ', response_text)
            #token_count = response['prompt_eval_count']
            #print(f"Real n tokens in prompt: {token_count}")

            prompt_selfcheck = f"""Ты — специалист по категоризации телефонных разговоров.
Есть записи телефонных разговоров менеджеров с клиентами, которые берут в аренду грязезащитные ковры и получают услуги по их доставке (замене) и чистке.

Вот текст одного разговора:
{truncated_text}

Ты присвоил ему следующие теги:
{response_text}, которые ты выбрал из списка:
{', '.join(self.tags_list)}

Проверь себя! Точно ли каждый из выбранных тобой тегов отражает реальную проблему / причину обращения клиента, а не просто содержит те же ключевые слова, что встречаются в тексте разговора?
Не забыл ли ты добавить какие-нибудь теги?
Если нужно - исправь свой ответ. Если не нашел неточностей - оставь ответ тем же.
Выбирай тег консультация_или_уточнение_деталей, ТОЛЬКО если нет никакой другой причины обращения!

ВЕРНИ ОТВЕТ ТОЛЬКО В ФОРМАТЕ JSON (от 0 до 3 тегов):
{{
  "result": ["tag1", "tag2"]
}}
Если текст не содержит ясной причины обращения - верни пустой json
"""

        #             print('Getting response - try 2 ...')
        #             response = self.client.generate(
        #                 model=self.model_name,
        #                 prompt=prompt_selfcheck,
        #                 options={
        #                     'temperature': 0.3,
        #                     'top_p': 0.8,
        #                     'num_ctx': 3000
        #                 }
        #             )
        #             response_text = response['response']
        #             print('response (corrected): ', response_text)

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                valid_selected = []
                for tag in result.get('result', []):
                    if tag in self.tags_list:
                        valid_selected.append(tag)
                    else:
                        print(f"Модель придумала тег '{tag}', игнорирую")
                valid_selected.append(self.data_type.value)
                print(valid_selected)

                result['result'] = valid_selected
                return result
            else:
                raise ValueError("LLM не вернула JSON")

        except Exception as e:
            print(f"Ошибка при запросе к LLM: {e}")
            return empty_response
