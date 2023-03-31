# 导入基础包内容，openai基本配置
import config

import tiktoken
import openai
openai.api_key = config.openai_api_key


CHAT_MODES = config.chat_modes

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}


class ChatGPT:
    #从chat模式中调取，看是否存在这个模式
    def __init__(self, model="gpt-3.5-turbo"):
        assert model in {"text-davinci-003", "gpt-3.5-turbo", "gpt-4"}, f"Unknown model: {model}"
        self.model = model

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in CHAT_MODES.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

# 不断向api发送消息，直到收到响应。
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:

#如果开启使用use-chatgpt-api则调用此方法反馈回答作为消息回复
            try:
                if self.model in {"gpt-3.5-turbo", "gpt-4"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]

#如果开启使用use-chatgpt-api则调用此方法反馈回答作为消息回复


#如果不开启使用use-chatgpt-api（在其他文件中）则调用此方法反馈回答作为消息回复
                elif self.model == "text-davinci-003":
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r = await openai.Completion.acreate(
                        engine=self.model,
                        prompt=prompt,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].text
                else:
                    raise ValueError(f"Unknown model: {model}")

#将回答内容处理后转换为可读格式，并且取出token值，计入用户使用token的值
                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.prompt_tokens, r.usage.completion_tokens
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed


# 流式传输文本消息，支持异步迭代
#发送一个消息流。将传入的 message 参数作为起始消息发送，
# 然后根据接收到的回复继续发送消息，直到对话结束。
# 在整个对话过程中，函数维护一个 dialog_messages 列表，用于存储所有发出的消息和接收到的回复。
    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in CHAT_MODES.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")
        
        #使用turbo模型生成回复。        
        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"gpt-3.5-turbo", "gpt-4"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r_gen = await openai.ChatCompletion.acreate(
                        model=self.model,
                        messages=messages,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

#使用了一个异步迭代器来获取每个生成的部分回答，直到生成完整的回答为止。生成的回答被逐步添加到字符串“answer”中，并使用异步生成器逐步输出。在循环结束时，将"not_finished"和最终的回答一起返回。
          #两个模式都支持异步迭代
                    answer = ""
                    async for r_item in r_gen:
                        delta = r_item.choices[0].delta
                        if "content" in delta:
                            answer += delta.content
                            yield "not_finished", answer

  #存储token值，计入计数器中
                    n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                elif self.model == "text-davinci-003":
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r_gen = await openai.Completion.acreate(
                        engine=self.model,
                        prompt=prompt,
                        stream=True,
                        **OPENAI_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        answer += r_item.choices[0].text
                        yield "not_finished", answer

                    n_input_tokens, n_output_tokens = self._count_tokens_from_prompt(prompt, answer, model=self.model)

                answer = self._postprocess_answer(answer)
 #使用了一个异步迭代器来获取每个生成的部分回答，直到生成完整的回答为止。生成的回答被逐步添加到字符串“answer”中，并使用异步生成器逐步输出。在循环结束时，将"not_finished"和最终的回答一起返回。
           

        #如果api返回错误显示too many tokens则检查dialog

            except openai.error.InvalidRequestError as e:  # too many tokens
                 #检查历史会话
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

           # 删除历史对话的第一条消息
                dialog_messages = dialog_messages[1:]
#计算在调用 send_message_stream 函数之前 dialog_messages 列表中的消息数，减去函数执行完 API 请求后 dialog_messages 列表中的消息数，即删除的消息数。
        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
 #这行代码是用来返回生成的答案、使用的 token 数量以及在生成答案时去除了多少条对话消息。yield 语句返回一个生成器对象，每次迭代时生成器会暂停并返回一个元组，其中第一个元素表示是否已经完成生成答案，第二个元素是生成的答案字符串，第三个元素是使用的 token 数量，第四个元素是去除的对话消息数。
        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer
 #这行代码是用来返回生成的答案、使用的 token 数量以及在生成答案时去除了多少条对话消息。yield 语句返回一个生成器对象，每次迭代时生成器会暂停并返回一个元组，其中第一个元素表示是否已经完成生成答案，第二个元素是生成的答案字符串，第三个元素是使用的 token 数量，第四个元素是去除的对话消息数。
   
   

    #（相当于人设预设）生成 OpenAI GPT API 的 prompt 。prompt 是一个用于给模型提供输入的字符串。这个函数将用户发来的消息、之前对话中用户和机器人的消息以及所处的聊天模式（chat_mode）组合成一个新的 prompt 并返回。
    #prompt是为了将之前的对话历史和当前的用户输入拼接成一个完整的文本，作为GPT模型的输入。如果不使用prompt，就需要自己手动将历史对话和当前输入文本拼接起来，以一定格式传给GPT模型作为输入。
    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # 在生成对话历史的prompt，主要是将之前的对话历史加入到新的prompt中，以保证生成的回答与之前的对话相连贯，
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # 目前的对话
        prompt += f"User: {message}\n"#user指的是用户发送的消息，而ChatGPT指的是聊天机器人回复的消息。
        prompt += "Assistant: "

        return prompt
    
    # 在生成对话历史的prompt，主要是将之前的对话历史加入到新的prompt中，以保证生成的回答与之前的对话相连贯，
    #（相当于人设预设）生成 OpenAI GPT API 的 prompt 。prompt 是一个用于给模型提供输入的字符串。这个函数将用户发来的消息、之前对话中用户和机器人的消息以及所处的聊天模式（chat_mode）组合成一个新的 prompt 并返回。

 #接收新消息的同时遍历历史消息、对话模式。格式化成api支持的格式。根据CHAT_MODES[chat_mode]["prompt_start"]的值生成对话的开头部分，然后遍历对话历史中的每条消息，将用户和聊天机器人的消息分别转换成OpenAI Chat API所需的格式，最后将用户输入的消息也添加到消息列表中并返回。
    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"]

        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages
 #接收新消息的同时遍历历史消息、对话模式。格式化成api支持的格式。根据CHAT_MODES[chat_mode]["prompt_start"]的值生成对话的开头部分，然后遍历对话历史中的每条消息，将用户和聊天机器人的消息分别转换成OpenAI Chat API所需的格式，最后将用户输入的消息也添加到消息列表中并返回。

 #对机器人回答进行后处理，即去除字符串首尾的空白字符，然后返回处理后的字符串。
    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer
#对机器人回答进行后处理，即去除字符串首尾的空白字符，然后返回处理后的字符串。


#计算给定对话历史消息和ChatGPT回答后，总共使用的token数量。（新代码支持输入和输出分开）
    def _count_tokens_from_messages(self, messages, answer, model="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model)

        if model == "gpt-3.5-turbo":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens
#计算给定对话历史消息和ChatGPT回答后，总共使用的token数量。


#Python的异步函数，用于将音频文件转录成文本。
# 使用OpenAI的语音转录API（名为 "atranscribe"），
# 并将音频文件和转录结果作为参数和返回值。
# 使用了Python的 "await" 关键字来等待API的响应，并将响应的文本部分返回。
# "transcribe_audio" 函数的目的是将语音转录为可供计算机程序分析和处理的文本形式。
async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]