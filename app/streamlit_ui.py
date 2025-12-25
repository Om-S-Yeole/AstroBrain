"""
AstroBrain Streamlit UI
Beautiful web interface for interacting with the AstroBrain aerospace AI assistant.
Connects to the WebSocket backend for real-time agent communication.
"""

import base64
import json
import uuid
from datetime import datetime
from io import BytesIO

import streamlit as st
import websocket
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AstroBrain - Aerospace AI Assistant",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful UI
st.markdown(
    """
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #60a5fa;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --dark-bg: #0f172a;
        --light-bg: #f8fafc;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f1f5f9;
        color: #0f172a;
        margin-right: 2rem;
        border-left: 4px solid #3b82f6;
    }
    
    .clarification-message {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: white;
        border: 2px solid #d97706;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        50% { box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
    }
    
    .system-message {
        background: #e0e7ff;
        color: #1e3a8a;
        text-align: center;
        font-style: italic;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #fee2e2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-connected {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-disconnected {
        background: #fee2e2;
        color: #991b1b;
    }
    
    .status-processing {
        background: #dbeafe;
        color: #1e3a8a;
        animation: blink 1.5s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Agent badges */
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .agent-orbitqa {
        background: #dbeafe;
        color: #1e3a8a;
    }
    
    .agent-missionops {
        background: #fce7f3;
        color: #831843;
    }
    
    /.* Input area */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Plot container */
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #1e0926;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #17092b;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Spinner animation */
    .custom-spinner {
        border: 3px solid #e2e8f0;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "ws_connection" not in st.session_state:
        st.session_state.ws_connection = None
    if "is_connected" not in st.session_state:
        st.session_state.is_connected = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False
    if "clarification_message" not in st.session_state:
        st.session_state.clarification_message = ""
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = None
    if "ws_url" not in st.session_state:
        st.session_state.ws_url = "ws://localhost:8000"
    if "connection_error" not in st.session_state:
        st.session_state.connection_error = None
    if "received_messages" not in st.session_state:
        st.session_state.received_messages = []


class WebSocketClient:
    """Handles WebSocket communication with the AstroBrain backend."""

    def __init__(self, url: str, thread_id: str):
        self.url = f"{url}/ws/{thread_id}"
        self.thread_id = thread_id
        self.ws = None
        self.is_connected = False
        self.message_queue = []

    def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            self.ws = websocket.create_connection(self.url, timeout=10)
            self.is_connected = True
            st.session_state.connection_error = None
            return True
        except Exception as e:
            self.is_connected = False
            st.session_state.connection_error = f"Connection failed: {str(e)}"
            return False

    def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.is_connected = False

    def send_query(self, message: str) -> bool:
        """Send a query message to the backend."""
        if not self.is_connected:
            return False

        try:
            data = {"type": "query", "message": message}
            self.ws.send(json.dumps(data))
            return True
        except (
            websocket.WebSocketConnectionClosedException,
            ConnectionResetError,
            BrokenPipeError,
            OSError,
        ) as e:
            st.session_state.connection_error = f"Send failed: {str(e)}"
            self.is_connected = False
            return False
        except Exception as e:
            # Other errors shouldn't disconnect
            st.session_state.connection_error = f"Send error: {str(e)}"
            return False

    def send_clarification(self, message: str) -> bool:
        """Send a clarification response to the backend."""
        if not self.is_connected:
            return False

        try:
            data = {"type": "clarification", "message": message}
            self.ws.send(json.dumps(data))
            return True
        except (
            websocket.WebSocketConnectionClosedException,
            ConnectionResetError,
            BrokenPipeError,
            OSError,
        ) as e:
            st.session_state.connection_error = f"Send failed: {str(e)}"
            self.is_connected = False
            return False
        except Exception as e:
            # Other errors shouldn't disconnect
            st.session_state.connection_error = f"Send error: {str(e)}"
            return False

    def receive_message(self, timeout: float = 0.1) -> dict | None:
        """Receive a message from the backend (non-blocking)."""
        if not self.is_connected:
            return None

        try:
            self.ws.settimeout(timeout)
            msg = self.ws.recv()
            return json.loads(msg)
        except websocket.WebSocketTimeoutException:
            # Timeout is normal for non-blocking receive, connection is still alive
            return None
        except websocket.WebSocketConnectionClosedException:
            # Connection was explicitly closed by server
            st.session_state.connection_error = "Connection closed by server"
            self.is_connected = False
            return None
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # Network-level connection errors
            st.session_state.connection_error = f"Connection lost: {str(e)}"
            self.is_connected = False
            return None
        except Exception as e:
            # Other errors (like JSON decode errors) shouldn't disconnect
            # Just log the error and continue
            error_type = type(e).__name__
            if "connection" in error_type.lower() or "closed" in str(e).lower():
                st.session_state.connection_error = f"Connection error: {str(e)}"
                self.is_connected = False
            # For other errors, just skip this message but keep connection alive
            return None


def render_header():
    """Render the main header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🛰️ AstroBrain</h1>
            <p>Aerospace AI Assistant - Orbital Mechanics & Mission Analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with connection settings and status."""
    with st.sidebar:
        st.markdown("### 🔧 Connection Settings")

        # WebSocket URL input
        ws_url = st.text_input(
            "WebSocket Server URL",
            value=st.session_state.ws_url,
            help="Enter the WebSocket server URL (e.g., ws://localhost:8000)",
        )

        if ws_url != st.session_state.ws_url:
            st.session_state.ws_url = ws_url
            if st.session_state.is_connected:
                st.warning("URL changed. Please reconnect.")

        # Connection button
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🔌 Connect", use_container_width=True, disabled=st.session_state.is_connected
            ):
                connect_to_backend()

        with col2:
            if st.button(
                "🔌 Disconnect",
                use_container_width=True,
                disabled=not st.session_state.is_connected,
            ):
                disconnect_from_backend()

        # Connection status
        st.markdown("---")
        st.markdown("### 📊 Status")

        if st.session_state.is_connected:
            st.markdown(
                '<span class="status-badge status-connected">✓ Connected</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge status-disconnected">✗ Disconnected</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.is_processing:
            st.markdown(
                '<span class="status-badge status-processing">⚙️ Processing...</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.current_agent:
            agent_class = (
                "agent-orbitqa"
                if st.session_state.current_agent == "orbitqa"
                else "agent-missionops"
            )
            agent_name = "OrbitQA" if st.session_state.current_agent == "orbitqa" else "MissionOps"
            st.markdown(
                f'<span class="agent-badge {agent_class}">🤖 {agent_name}</span>',
                unsafe_allow_html=True,
            )

        # Display connection error if any
        if st.session_state.connection_error:
            st.error(st.session_state.connection_error)

        # Session info
        st.markdown("---")
        st.markdown("### ℹ️ Session Info")
        st.code(f"Thread ID: {st.session_state.thread_id[:8]}...", language="text")

        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.awaiting_clarification = False
            st.session_state.clarification_message = ""
            st.session_state.current_agent = None
            st.rerun()

        # About section
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            """
            **AstroBrain** is an AI-powered aerospace assistant that helps with:
            
            - 🛰️ Orbital mechanics calculations
            - 📊 Mission feasibility analysis
            - 📈 Orbit propagation & visualization
            - 🔋 Power, thermal & communication analysis
            - ☀️ Eclipse & sun geometry calculations
            
            **Available Agents:**
            - **OrbitQA**: Quick orbital calculations & plots
            - **MissionOps**: Comprehensive mission analysis
            """
        )


def connect_to_backend():
    """Establish connection to the WebSocket backend."""
    try:
        client = WebSocketClient(st.session_state.ws_url, st.session_state.thread_id)
        if client.connect():
            st.session_state.ws_connection = client
            st.session_state.is_connected = True
            st.success("✓ Connected to AstroBrain backend!")
            st.rerun()
        else:
            st.error("Failed to connect to backend. Please check the server URL.")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")


def disconnect_from_backend():
    """Disconnect from the WebSocket backend."""
    if st.session_state.ws_connection:
        st.session_state.ws_connection.disconnect()
        st.session_state.ws_connection = None
    st.session_state.is_connected = False
    st.session_state.is_processing = False
    st.session_state.awaiting_clarification = False
    st.info("Disconnected from backend.")
    st.rerun()


def handle_incoming_messages():
    """Handle incoming messages from the WebSocket."""
    if not st.session_state.is_connected or not st.session_state.ws_connection:
        return

    # Sync connection state from WebSocket client
    if not st.session_state.ws_connection.is_connected:
        st.session_state.is_connected = False
        st.session_state.is_processing = False
        return

    # Check for new messages
    while True:
        msg = st.session_state.ws_connection.receive_message(timeout=0.1)
        if msg is None:
            break

        msg_type = msg.get("type")

        if msg_type == "status":
            # Status update from backend
            status_msg = msg.get("message", "")
            agent = msg.get("agent")

            if agent:
                st.session_state.current_agent = agent

            # Add system message
            st.session_state.messages.append(
                {"role": "system", "content": status_msg, "timestamp": datetime.now()}
            )

        elif msg_type == "clarification_request":
            # Agent needs clarification
            clarification_msg = msg.get("message", "")
            st.session_state.awaiting_clarification = True
            st.session_state.clarification_message = clarification_msg
            st.session_state.is_processing = False

            # Add clarification request to messages
            st.session_state.messages.append(
                {
                    "role": "clarification",
                    "content": clarification_msg,
                    "timestamp": datetime.now(),
                }
            )

        elif msg_type == "response":
            # Final response from agent
            st.session_state.is_processing = False
            response_data = msg.get("data", {})
            final_response = response_data.get("final_response")

            if final_response:
                # Add assistant response
                assistant_msg = {
                    "role": "assistant",
                    "content": final_response.get("message", ""),
                    "status": final_response.get("status", ""),
                    "plots": final_response.get("plots", []),
                    "warnings": final_response.get("warnings", []),
                    "timestamp": datetime.now(),
                }
                st.session_state.messages.append(assistant_msg)

        elif msg_type == "error":
            # Error from backend
            st.session_state.is_processing = False
            error_msg = msg.get("message", "Unknown error")
            st.session_state.messages.append(
                {"role": "error", "content": error_msg, "timestamp": datetime.now()}
            )

        elif msg_type == "pong":
            # Keep-alive response
            pass


def render_message(msg: dict):
    """Render a single chat message."""
    role = msg.get("role")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", datetime.now())

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-message user-message">
                <strong>You</strong> <small>({timestamp.strftime("%H:%M:%S")})</small><br>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif role == "assistant":
        warnings = msg.get("warnings", [])
        plots = msg.get("plots", [])

        st.markdown(
            f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AstroBrain</strong> <small>({timestamp.strftime("%H:%M:%S")})</small><br>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display warnings
        if warnings:
            for warning in warnings:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        ⚠️ <strong>Warning:</strong> {warning}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Display plots
        if plots:
            for i, plot_base64 in enumerate(plots):
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(plot_base64)
                    image = Image.open(BytesIO(image_data))

                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    st.image(image, caption=f"Plot {i + 1}", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying plot: {str(e)}")

    elif role == "clarification":
        st.markdown(
            f"""
            <div class="chat-message clarification-message">
                <strong>🔔 Clarification Needed</strong> <small>({timestamp.strftime("%H:%M:%S")})</small><br>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif role == "system":
        st.markdown(
            f"""
            <div class="system-message">
                ℹ️ {content}
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif role == "error":
        st.markdown(
            f"""
            <div class="error-message">
                <strong>❌ Error</strong><br>
                {content}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_interface():
    """Render the main chat interface."""
    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display all messages
        for msg in st.session_state.messages:
            render_message(msg)

        # Show processing indicator
        if st.session_state.is_processing:
            st.markdown(
                """
                <div class="chat-message assistant-message">
                    <strong>🤖 AstroBrain</strong><br>
                    <div class="custom-spinner"></div>
                    Processing your query...
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Input area
    st.markdown("---")

    if not st.session_state.is_connected:
        st.warning("⚠️ Please connect to the backend server first using the sidebar.")
        st.stop()

    if st.session_state.awaiting_clarification:
        # Clarification input
        st.markdown(
            """
            <div class="info-box">
                💬 <strong>Please provide clarification:</strong><br>
                The agent needs additional information to proceed.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form(key="clarification_form", clear_on_submit=True):
            clarification_input = st.text_input(
                "Your clarification:",
                placeholder="Type your response here...",
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button("📤 Send Clarification", use_container_width=True)

            if submit_button and clarification_input.strip():
                # Send clarification
                if st.session_state.ws_connection.send_clarification(clarification_input):
                    # Add user message
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": clarification_input,
                            "timestamp": datetime.now(),
                        }
                    )
                    st.session_state.awaiting_clarification = False
                    st.session_state.is_processing = True
                    st.rerun()
                else:
                    st.error("Failed to send clarification. Please check connection.")

    else:
        # Regular query input
        with st.form(key="query_form", clear_on_submit=True):
            user_input = st.text_input(
                "Your query:",
                placeholder="Ask me about orbital mechanics, mission analysis, or anything aerospace-related...",
                disabled=st.session_state.is_processing,
                label_visibility="collapsed",
            )
            submit_button = st.form_submit_button(
                "🚀 Send Query",
                use_container_width=True,
                disabled=st.session_state.is_processing,
            )

            if submit_button and user_input.strip():
                # Send query
                if st.session_state.ws_connection.send_query(user_input):
                    # Add user message
                    st.session_state.messages.append(
                        {"role": "user", "content": user_input, "timestamp": datetime.now()}
                    )
                    st.session_state.is_processing = True
                    st.rerun()
                else:
                    st.error("Failed to send query. Please check connection.")


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Render UI components
    render_header()
    render_sidebar()

    # Handle incoming messages
    if st.session_state.is_connected:
        handle_incoming_messages()

    # Render chat interface
    render_chat_interface()

    # Auto-refresh while processing or awaiting clarification
    if st.session_state.is_processing or st.session_state.awaiting_clarification:
        import time

        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
