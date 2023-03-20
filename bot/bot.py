#导入基本包
import os
import logging
import asyncio
import traceback
import html
import json
import tempfile
import pydub
from pathlib import Path
from datetime import datetime
import telegram
from telegram import (
    Update, 
    User, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup, 
    BotCommand
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    AIORateLimiter,
    filters
)
from telegram.constants import ParseMode, ChatAction

import config
import database
import openai_utils

#导入基本包


# 安装
db = database.Database()
logger = logging.getLogger(__name__)
user_semaphores = {}
#指令菜单

HELP_MESSAGE = """告诉我我要干嘛：
⚪ /retry – 重新生成上一个回答
⚪ /new – 我要开始新对话
⚪ /mode – 我要你更专注于
⚪ /balance – 看看翼臣哥哥的钱包
⚪ /help – 我需要帮助
"""

#欢迎内容
def split_text_into_chunks(text, chunk_size):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


async def register_user_if_not_exists(update: Update, context: CallbackContext, user: User):
    if not db.check_if_user_exists(user.id):
        db.add_new_user(
            user.id,
            update.message.chat_id,
            username=user.username,
            first_name=user.first_name,
            last_name= user.last_name
        )
        db.start_new_dialog(user.id)

    if db.get_user_attribute(user.id, "current_dialog_id") is None:
        db.start_new_dialog(user.id)

    if user.id not in user_semaphores:
        user_semaphores[user.id] = asyncio.Semaphore(1)


async def start_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    db.start_new_dialog(user_id)
    
    reply_text = "你好，我是基于GPT-3.5 OpenAI API的<b>ChatGPT</b> 机器人，我是翼 🤖\n\n"
    reply_text += HELP_MESSAGE

    reply_text += "\n现在...问我任何事!"
    
    await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)

#欢迎内容

#帮助指令
async def help_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())
    await update.message.reply_text(HELP_MESSAGE, parse_mode=ParseMode.HTML)
#帮助指令

#重试指令
async def retry_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return
    
    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
    if len(dialog_messages) == 0:
        await update.message.reply_text("你还没有发送消息呢")
        return

    last_dialog_message = dialog_messages.pop()
    db.set_dialog_messages(user_id, dialog_messages, dialog_id=None)  # last message was removed from the context

    await message_handle(update, context, message=last_dialog_message["user"], use_new_dialog_timeout=False)
##重试指令

#消息处理
async def message_handle(update: Update, context: CallbackContext, message=None, use_new_dialog_timeout=True):
    # check if message is edited
    if update.edited_message is not None:
        await edited_message_handle(update, context)
        return
        
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    #消息对话超时自动生成新对话
    async with user_semaphores[user_id]:
        # new dialog timeout
        if use_new_dialog_timeout:
            if (datetime.now() - db.get_user_attribute(user_id, "last_interaction")).seconds > config.new_dialog_timeout and len(db.get_dialog_messages(user_id)) > 0:
                db.start_new_dialog(user_id)
                await update.message.reply_text(f"由于上个对话时间超时,现在是新对话 (<b>{openai_utils.CHAT_MODES[chat_mode]['name']}</b>) ✅", parse_mode=ParseMode.HTML)
        db.set_user_attribute(user_id, "last_interaction", datetime.now())

        # 发送消息格式类型设置
        await update.message.chat.send_action(action="typing")

        try:
            message = message or update.message.text

            dialog_messages = db.get_dialog_messages(user_id, dialog_id=None)
            parse_mode = {
                "html": ParseMode.HTML,
                "markdown": ParseMode.MARKDOWN
            }[openai_utils.CHAT_MODES[chat_mode]["parse_mode"]]

            chatgpt_instance = openai_utils.ChatGPT(use_chatgpt_api=config.use_chatgpt_api)
            if config.enable_message_streaming:
                gen = chatgpt_instance.send_message_stream(message, dialog_messages=dialog_messages, chat_mode=chat_mode)
            else:
                answer, n_used_tokens, n_first_dialog_messages_removed = await chatgpt_instance.send_message(
                    message,
                    dialog_messages=dialog_messages,
                    chat_mode=chat_mode
                )

                async def fake_gen():
                    yield "finished", answer, n_used_tokens, n_first_dialog_messages_removed

                gen = fake_gen()

            #将信息发送给用户
            prev_answer = ""
            i = -1
            async for gen_item in gen:
                i += 1

                status = gen_item[0]
                if status == "not_finished":
                    status, answer = gen_item
                elif status == "finished":
                    status, answer, n_used_tokens, n_first_dialog_messages_removed = gen_item
                else:
                    raise ValueError(f"Streaming status {status} is unknown")

                answer = answer[:4096]  # telegram message limit
                if i == 0:  # send first message (then it'll be edited if message streaming is enabled)
                    try:                    
                        sent_message = await update.message.reply_text(answer, parse_mode=parse_mode)
                    except telegram.error.BadRequest as e:
                        if str(e).startswith("Message must be non-empty"):  # first answer chunk from openai was empty
                            i = -1  # try again to send first message
                            continue
                        else:
                            sent_message = await update.message.reply_text(answer)
                else:  # edit sent message
                    # update only when 100 new symbols are ready
                    if abs(len(answer) - len(prev_answer)) < 100 and status != "finished":
                        continue

                    try:                    
                        await context.bot.edit_message_text(answer, chat_id=sent_message.chat_id, message_id=sent_message.message_id, parse_mode=parse_mode)
                    except telegram.error.BadRequest as e:
                        if str(e).startswith("Message is not modified"):
                            continue
                        else:
                            await context.bot.edit_message_text(answer, chat_id=sent_message.chat_id, message_id=sent_message.message_id)

                    await asyncio.sleep(0.01)  # 等待一会而避免flooding
                    
                prev_answer = answer

            # 更新用户数据
            new_dialog_message = {"user": message, "bot": answer, "date": datetime.now()}
            db.set_dialog_messages(
                user_id,
                db.get_dialog_messages(user_id, dialog_id=None) + [new_dialog_message],
                dialog_id=None
            )

          #进行流式传输时未得到回复出现错误提示用户
            db.set_user_attribute(user_id, "n_used_tokens", n_used_tokens + db.get_user_attribute(user_id, "n_used_tokens"))
        except Exception as e:
            error_text = f"你还没有回复上条消息呢,原因: {e}"
            logger.error(error_text)
            await update.message.reply_text(error_text)
            return
        
        #当信息从文本移除时发送提示消息给用户告知对方第一条消息被移除
 
        if n_first_dialog_messages_removed > 0:
            if n_first_dialog_messages_removed == 1:
                text = "✍️ <i>Note:</i> 你目前的对话太久了,因此你的<b>第一条消息</b> 从对话记录中被移除\n 发送 /new 命令创建新对话"
            else:
                text = f"✍️ <i>Note:</i>  你目前的对话太久了,因此此次对话的<b>{n_first_dialog_messages_removed} 第一条</b> 被移除. \n 发送 /new 命令开始新对话"
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        #当信息从文本移除时发送提示消息给用户告知对方第一条消息被移除



#当用户发送的消息未得到相应时回复
#如果该用户的信号量已经被锁定（即在先前的消息尚未被响应之前），则该函数将向用户发送一条提示消息，并返回True。 否则，函数将返回False，表示该用户可以发送一条新消息。
async def is_previous_message_not_answered_yet(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    if user_semaphores[user_id].locked():
        text = "⏳ 请<b>等待</b> 一个当前信息的回复"
        await update.message.reply_text(text, reply_to_message_id=update.message.id, parse_mode=ParseMode.HTML)
        return True
    else:
        return False

#当用户发送的消息未得到相应时回复


#声音消息处理
async def voice_message_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    voice = update.message.voice
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        voice_ogg_path = tmp_dir / "voice.ogg"

        # download
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_ogg_path)

        # convert to mp3
        voice_mp3_path = tmp_dir / "voice.mp3"
        pydub.AudioSegment.from_file(voice_ogg_path).export(voice_mp3_path, format="mp3")

        # transcribe
        with open(voice_mp3_path, "rb") as f:
            transcribed_text = await openai_utils.transcribe_audio(f)

    text = f"🎤: <i>{transcribed_text}</i>"
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    await message_handle(update, context, message=transcribed_text)

    # calculate spent dollars
    n_spent_dollars = voice.duration * (config.whisper_price_per_1_min / 60)

    # normalize dollars to tokens (it's very convenient to measure everything in a single unit)
    price_per_1000_tokens = config.chatgpt_price_per_1000_tokens if config.use_chatgpt_api else config.gpt_price_per_1000_tokens
    n_used_tokens = int(n_spent_dollars / (price_per_1000_tokens / 1000))
    db.set_user_attribute(user_id, "n_used_tokens", n_used_tokens + db.get_user_attribute(user_id, "n_used_tokens"))

 #声音消息处理



#启动新对话、获取用户名、使用的聊天模式
async def new_dialog_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    db.start_new_dialog(user_id)
    await update.message.reply_text("开始新对话 ✅")

    chat_mode = db.get_user_attribute(user_id, "current_chat_mode")
    await update.message.reply_text(f"{openai_utils.CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)
#启动新对话、获取用户名、使用的聊天模式

#展示模式内容
async def show_chat_modes_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)
    if await is_previous_message_not_answered_yet(update, context): return

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    keyboard = []
    for chat_mode, chat_mode_dict in openai_utils.CHAT_MODES.items():
        keyboard.append([InlineKeyboardButton(chat_mode_dict["name"], callback_data=f"set_chat_mode|{chat_mode}")])
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text("选择更让我专注的模式吧", reply_markup=reply_markup)

#展示模式内容

#设置模式
async def set_chat_mode_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update.callback_query, context, update.callback_query.from_user)
    user_id = update.callback_query.from_user.id

    query = update.callback_query
    await query.answer()

    chat_mode = query.data.split("|")[1]

    db.set_user_attribute(user_id, "current_chat_mode", chat_mode)
    db.start_new_dialog(user_id)

    await query.edit_message_text(f"{openai_utils.CHAT_MODES[chat_mode]['welcome_message']}", parse_mode=ParseMode.HTML)
#设置模式


#余额菜单
async def show_balance_handle(update: Update, context: CallbackContext):
    await register_user_if_not_exists(update, context, update.message.from_user)

    user_id = update.message.from_user.id
    db.set_user_attribute(user_id, "last_interaction", datetime.now())

    n_used_tokens = db.get_user_attribute(user_id, "n_used_tokens")

    price_per_1000_tokens = config.chatgpt_price_per_1000_tokens if config.use_chatgpt_api else config.gpt_price_per_1000_tokens
    n_spent_dollars = n_used_tokens * (price_per_1000_tokens / 1000)

    text = f"翼臣哥哥花费了 <b>{n_spent_dollars:.03f}$</b>\n"
    text += f"总共使用了 <b>{n_used_tokens}</b> tokens\n\n"

    text += "🏷️ 这是官方API价格,我也要成本的！！！\n"
    text += f"<i>- ChatGPT: {price_per_1000_tokens}$ per 1000 tokens\n"
    text += f"- Whisper (voice recognition): {config.whisper_price_per_1_min}$ per 1 minute</i>"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)
#余额菜单


#编辑后的消息文本设置反馈
async def edited_message_handle(update: Update, context: CallbackContext):
    text = "🥲sorry啦, <b>编辑</b> 后不支持,你需要重新发送新的内容而不是编辑旧内容。"
    await update.edited_message.reply_text(text, parse_mode=ParseMode.HTML)
#编辑后的消息文本设置反馈

#错误菜单
async def error_handle(update: Update, context: CallbackContext) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    try:
        # collect error message
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
            "</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )

        # split text into multiple messages due to 4096 character limit
        for message_chunk in split_text_into_chunks(message, 4096):
            try:
                await context.bot.send_message(update.effective_chat.id, message_chunk, parse_mode=ParseMode.HTML)
            except telegram.error.BadRequest:
                # answer has invalid characters, so we send it without parse_mode
                await context.bot.send_message(update.effective_chat.id, message_chunk)
    except:
        await context.bot.send_message(update.effective_chat.id, "Some error in error handler")
#错误


#机器人菜单界面
async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new", "我要开始新对话"),
        BotCommand("/mode", "我要你更专注"),
        BotCommand("/retry", "重新回答这个问题"),
        BotCommand("/balance", "看看翼臣哥哥的钱包"),
        BotCommand("/help", "我需要帮助"),
    ])
#机器人菜单界面

#运行机器人
def run_bot() -> None:
    application = (
        ApplicationBuilder()
        .token(config.telegram_token)
        .concurrent_updates(True)
        .rate_limiter(AIORateLimiter(max_retries=5))
        .post_init(post_init)
        .build()
    )

    # 对应菜单指令的处理程序
    user_filter = filters.ALL
    if len(config.allowed_telegram_usernames) > 0:
        usernames = [x for x in config.allowed_telegram_usernames if isinstance(x, str)]
        user_ids = [x for x in config.allowed_telegram_usernames if isinstance(x, int)]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids)

    application.add_handler(CommandHandler("start", start_handle, filters=user_filter))
    application.add_handler(CommandHandler("help", help_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & user_filter, message_handle))
    application.add_handler(CommandHandler("retry", retry_handle, filters=user_filter))
    application.add_handler(CommandHandler("new", new_dialog_handle, filters=user_filter))

    application.add_handler(MessageHandler(filters.VOICE & user_filter, voice_message_handle))
    
    application.add_handler(CommandHandler("mode", show_chat_modes_handle, filters=user_filter))
    application.add_handler(CallbackQueryHandler(set_chat_mode_handle, pattern="^set_chat_mode"))

    application.add_handler(CommandHandler("balance", show_balance_handle, filters=user_filter))
    
    application.add_error_handler(error_handle)
    
    # 启动
    application.run_polling()
#运行机器人

#设置特定用户启动
if __name__ == "__main__":
    run_bot()
