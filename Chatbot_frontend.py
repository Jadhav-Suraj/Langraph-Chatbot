from Chatbot_backend import chatbot, get_all_threads, ingest_rag_document, get_retriever
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command          # <-- NEW (HITL)
import streamlit as st
import uuid
import tempfile
import os
import shutil


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    add_thread(st.session_state['thread_id'])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def get_thread_title(thread_id):
    messages = load_conversation(thread_id)
    for message in messages:
        if isinstance(message, HumanMessage):
            title = message.content.strip()
            if len(title) > 15:
                return title[:15] + "..."
            return title
    return "New Chat"


def render_tools(tools):
    """Show which tools were used for an assistant message (history view)."""
    if tools:
        with st.status(f"🔧 Tool used: {', '.join(tools)}", state="complete", expanded=False):
            for t in tools:
                st.write(f"• `{t}`")


# saves the uploaded PDF temporarily (with its real name) and indexes it
def process_uploaded_pdf(uploaded_file):
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return ingest_rag_document(file_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------- HITL helpers (NEW) ----------------
def get_config(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace"
    }


def get_pending_interrupt(thread_id):
    """Return the interrupt prompt if this thread is paused waiting for a human, else None."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def stream_graph(payload):
    """
    Streams the graph for either a normal input ({'messages': [...]})
    or a resume command (Command(resume='yes'/'no')).
    Returns (final_text, tools_used).
    """
    CONFIG = get_config(st.session_state['thread_id'])

    with st.chat_message("assistant"):
        tool_area = st.container()   # tool status appears ABOVE the answer text
        used_tools = []
        status_box = {"box": None}

        def stream_response():
            for message_chunk, metadata in chatbot.stream(
                payload,
                config=CONFIG,
                stream_mode='messages'
            ):
                # 1) LLM decided to call a tool
                if isinstance(message_chunk, AIMessage):
                    for tc in (message_chunk.tool_call_chunks or []):
                        name = tc.get("name")
                        if name:
                            used_tools.append(name)
                            with tool_area:
                                if status_box["box"] is None:
                                    status_box["box"] = st.status(
                                        f"🔧 Using tool: `{name}` ...", expanded=True
                                    )
                                else:
                                    status_box["box"].update(
                                        label=f"🔧 Using tool: `{name}` ...",
                                        state="running", expanded=True
                                    )

                # 2) Tool finished and returned a result
                if isinstance(message_chunk, ToolMessage):
                    if status_box["box"] is not None:
                        result = str(message_chunk.content)
                        status_box["box"].write(f"**`{message_chunk.name}`** returned:")
                        status_box["box"].text(result[:500] + ("..." if len(result) > 500 else ""))

                # 3) Normal answer text
                if isinstance(message_chunk, AIMessage) and message_chunk.text:
                    yield message_chunk.text

        ai_message = st.write_stream(stream_response())

        # collapse the status box when the answer is done
        if status_box["box"] is not None:
            # if graph is paused for approval, show "waiting" instead of "done"
            if get_pending_interrupt(st.session_state['thread_id']):
                status_box["box"].update(
                    label=f"⏸️ Waiting for approval: {', '.join(used_tools)}",
                    state="running", expanded=False
                )
            else:
                status_box["box"].update(
                    label=f"✅ Tool used: {', '.join(used_tools)}",
                    state="complete", expanded=False
                )

    return ai_message, used_tools


st.title("Agentic Chatbot")

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'indexed_file' not in st.session_state:
    st.session_state['indexed_file'] = None     # stores "name-size" key

if 'indexed_name' not in st.session_state:
    st.session_state['indexed_name'] = None     # stores file name to display

add_thread(st.session_state['thread_id'])

# ------------------------------- SideBar ------------------

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

# ------------------------------- Document status only ------------------
st.sidebar.divider()

if st.session_state['indexed_name']:
    st.sidebar.caption(f"📄 Active document: **{st.session_state['indexed_name']}**")
elif os.path.exists("faiss_db"):
    st.sidebar.caption("📄 A previously indexed document is available.")
else:
    st.sidebar.caption("📄 No document uploaded yet. Use the + button in the chat box.")

st.sidebar.divider()
# ------------------------------------------------------------------------

for thread_id in st.session_state['chat_threads'][::-1]:

    chat_title = get_thread_title(thread_id)

    if st.sidebar.button(chat_title, key=thread_id, use_container_width=True):

        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        pending_tools = []   # tools called before the next final AI answer

        for message in messages:

            if isinstance(message, HumanMessage):
                temp_messages.append({"role": "user", "content": message.content})

            elif isinstance(message, AIMessage):
                # collect tool names this AI message asked for
                for tc in (message.tool_calls or []):
                    pending_tools.append(tc["name"])

                # only keep AI messages that actually have text (final answer)
                if message.text:
                    temp_messages.append({
                        "role": "assistant",
                        "content": message.text,
                        "tools": pending_tools,
                    })
                    pending_tools = []

            # ToolMessage etc. are skipped

        st.session_state['message_history'] = temp_messages
        st.rerun()

## loading conversation history from session state
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            render_tools(message.get('tools'))
        st.text(message['content'])


# ------------------------------- HITL: Approval card (NEW) ------------------
pending_prompt = get_pending_interrupt(st.session_state['thread_id'])

if pending_prompt:
    approval_slot = st.empty()
    with approval_slot.container(border=True):
        st.warning("⚠️ **Human approval required**")
        st.markdown(f"**{pending_prompt}**")
        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve", key="hitl_approve", use_container_width=True, type="primary")
        reject = col2.button("❌ Reject", key="hitl_reject", use_container_width=True)

    if approve or reject:
        decision = "yes" if approve else "no"
        approval_slot.empty()     # remove the card immediately

        ai_message, used_tools = stream_graph(Command(resume=decision))

        st.session_state['message_history'].append({
            'role': 'assistant',
            'content': ai_message,
            'tools': used_tools,
        })
        st.rerun()                # refresh state (re-enables chat input / handles another interrupt)


# ------------------------------- Chat input with built-in PDF attach ------------------
# accept_file=True adds a "+" attach button inside the chat box. Needs Streamlit >= 1.43.
# Disabled while waiting for human approval.
chat_value = st.chat_input(
    "Approve or reject the pending action first..." if pending_prompt else "Type Here",
    accept_file=True,
    file_type=["pdf"],
    disabled=bool(pending_prompt),
)

user_input = None     # the typed text
uploaded_pdf = None   # the attached PDF, if any

if chat_value:
    user_input = chat_value.text
    uploaded_pdf = chat_value.files[0] if chat_value.files else None

# ------------------------------- PDF indexing (unchanged) ------------------
if uploaded_pdf is not None:
    file_key = f"{uploaded_pdf.name}-{uploaded_pdf.size}"

    # index only once per file (Streamlit reruns the script on every interaction)
    if st.session_state['indexed_file'] != file_key:
        with st.spinner("Indexing document..."):
            try:
                n_chunks = process_uploaded_pdf(uploaded_pdf)
                if n_chunks:
                    st.session_state['indexed_file'] = file_key
                    st.session_state['indexed_name'] = uploaded_pdf.name
                    st.toast(f"Indexed {n_chunks} chunks ✅")
                else:
                    st.toast("No text found in this PDF (it may be scanned).", icon="⚠️")
            except Exception as e:
                st.toast(f"Failed to index PDF: {e}", icon="❌")

# ------------------------------- Chat logic ------------------
# Runs only when the user typed a message. If they attached a PDF with no text,
# the PDF is just indexed above and no chat call is made.
if user_input:
    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    with st.chat_message("user"):
        st.text(user_input)

    ai_message, used_tools = stream_graph({'messages': [HumanMessage(content=user_input)]})

    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_message,
        'tools': used_tools,
    })

    # if the graph paused (e.g. purchase_stock), rerun so the approval card appears
    if get_pending_interrupt(st.session_state['thread_id']):
        st.rerun()