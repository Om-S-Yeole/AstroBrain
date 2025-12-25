# 🛰️ AstroBrain UI - Quick Start Guide

Welcome to AstroBrain's Streamlit UI! This guide will help you get started with the beautiful web interface for interacting with the AstroBrain aerospace AI assistant.

## 📋 Prerequisites

Before running the UI, make sure you have:

1. **Python Environment**: Activated virtual environment (`myenv`)
2. **Required Packages**: Streamlit and websocket-client installed
3. **Backend Server**: WebSocket server running on `localhost:8000`

## 🚀 Installation

### Step 1: Activate Virtual Environment

```powershell
# Navigate to project directory
cd C:\Users\hp\OneDrive\Documents\Projects\AstroBrain

# Activate the virtual environment
.\myenv\Scripts\Activate.ps1
```

### Step 2: Install Required Packages

```powershell
# Install Streamlit and websocket-client
pip install streamlit websocket-client pillow
```

### Step 3: Start the WebSocket Backend Server

In a **separate terminal**, start the backend server:

```powershell
# Activate virtual environment
.\myenv\Scripts\Activate.ps1

# Start the WebSocket server
python -m app.websocket_server
```

The server should start on `http://localhost:8000`. You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Launch the Streamlit UI

In another terminal (or the original one), launch the UI:

```powershell
# Make sure virtual environment is activated
.\myenv\Scripts\Activate.ps1

# Launch Streamlit UI
streamlit run app/streamlit_ui.py
```

The browser should automatically open at `http://localhost:8501`.

## 🎨 Using the UI

### 1. **Connect to Backend**

- Click the **🔌 Connect** button in the sidebar
- The status should change to **✓ Connected**
- If connection fails, check that the backend server is running

### 2. **Ask Questions**

Type your aerospace-related questions in the input box at the bottom:

**Example Queries:**

**For OrbitQA Agent (Quick Calculations):**
- "Plot a circular orbit at 400 km altitude"
- "Calculate orbital velocity at 600 km altitude"
- "What is the orbital period for a 500 km altitude orbit?"
- "Show me a Molniya orbit"

**For MissionOps Agent (Mission Analysis):**
- "Analyze mission feasibility for a 500 km sun-synchronous orbit"
- "Evaluate power budget for a LEO satellite mission"
- "Assess thermal conditions for a satellite in 600 km orbit"
- "Check communication windows for a ground station at 40°N"

### 3. **Respond to Clarifications**

When the agent needs more information:

1. A **yellow clarification box** will appear with the agent's question
2. Type your answer in the clarification input field
3. Click **📤 Send Clarification**
4. The agent will continue processing with your input

**Example Clarification Flow:**
```
Agent: "What is the inclination angle in degrees?"
You: "98.5 degrees"
Agent: [Continues analysis with your input]
```

### 4. **View Results**

The agent's response will include:

- **📝 Text Response**: Detailed analysis and answers
- **📊 Plots**: Visual representations (orbits, graphs, etc.)
- **⚠️ Warnings**: Important notes or limitations
- **✅ Status**: Success or error indicators

### 5. **Monitor Status**

The sidebar shows:

- **Connection Status**: Connected/Disconnected
- **Processing Status**: When the agent is working
- **Current Agent**: Which agent (OrbitQA/MissionOps) is handling your query
- **Session ID**: Your unique session identifier

## 🎯 Features

### ✨ Beautiful Design

- **Gradient Headers**: Modern blue gradient design
- **Animated Messages**: Smooth fade-in animations
- **Status Badges**: Color-coded status indicators
- **Responsive Layout**: Works on different screen sizes

### 🔄 Real-Time Communication

- **WebSocket Connection**: Persistent, low-latency connection
- **Live Updates**: Messages appear instantly
- **Auto-Refresh**: UI updates automatically during processing

### 🛡️ Error Handling

- **Connection Errors**: Clear error messages with troubleshooting hints
- **Timeout Handling**: Automatic timeout detection
- **Graceful Degradation**: Continues working even after errors

### 💬 Clarification Support

- **Interactive Prompts**: Agent can ask follow-up questions
- **Visual Indicators**: Pulsing yellow boxes for clarifications
- **Seamless Flow**: Natural conversation-like interaction

### 📊 Rich Content Display

- **Image Plots**: Base64-decoded plots displayed inline
- **Formatted Messages**: Clean, readable message formatting
- **Warnings**: Highlighted warning messages
- **System Messages**: Status updates shown in context

## 🔧 Sidebar Controls

### Connection Settings

- **WebSocket Server URL**: Change the backend server address (default: `ws://localhost:8000`)
- **Connect/Disconnect**: Manage connection to backend

### Session Management

- **Clear Chat**: Reset the conversation history
- **Session Info**: View your unique session ID

### Status Indicators

- **Connection Status**: 
  - 🟢 **Connected**: Ready to use
  - 🔴 **Disconnected**: Not connected to backend
  
- **Processing Status**:
  - ⚙️ **Processing**: Agent is working on your query
  
- **Agent Badge**:
  - 🔵 **OrbitQA**: Handling orbital calculations
  - 🟣 **MissionOps**: Handling mission analysis

## 🐛 Troubleshooting

### Problem: Cannot Connect to Backend

**Solution:**
1. Verify the backend server is running:
   ```powershell
   # Check if server is running
   curl http://localhost:8000/health
   ```
2. Check the WebSocket URL in the sidebar (should be `ws://localhost:8000`)
3. Restart both backend and UI

### Problem: UI Not Responding

**Solution:**
1. Check browser console for errors (F12)
2. Refresh the page (Ctrl + R)
3. Clear browser cache
4. Restart Streamlit

### Problem: Messages Not Appearing

**Solution:**
1. Check connection status in sidebar
2. Look for error messages
3. Disconnect and reconnect
4. Check backend server logs

### Problem: Plots Not Displaying

**Solution:**
1. Ensure PIL/Pillow is installed: `pip install pillow`
2. Check browser console for image loading errors
3. Verify base64 data is valid in backend response

### Problem: Clarification Not Working

**Solution:**
1. Ensure you're typing in the clarification input (yellow box)
2. Click "Send Clarification" button
3. Check that the message was sent (watch for status update)
4. Wait for agent to process (status will show "Processing")

## 📝 Tips & Best Practices

### 1. **Be Specific in Queries**
- ✅ Good: "Calculate orbital velocity for a circular orbit at 500 km altitude"
- ❌ Vague: "Tell me about orbits"

### 2. **Provide Units**
- Always specify units (km, degrees, kg, etc.)
- Use standard aerospace conventions

### 3. **Use Appropriate Agent**
The system automatically routes your query, but you can guide it:
- Use **"calculate"**, **"plot"**, **"show"** for OrbitQA
- Use **"analyze"**, **"assess"**, **"feasibility"** for MissionOps

### 4. **Monitor Processing**
- Watch the sidebar for status updates
- Be patient during complex calculations
- Don't send multiple queries while processing

### 5. **Save Important Results**
- Take screenshots of plots
- Copy text responses before clearing chat
- Note session ID for troubleshooting

## 🎓 Example Session

Here's a complete example session:

```
[You connect to the backend]
Status: ✓ Connected

[You]: Plot a sun-synchronous orbit at 800 km altitude

Status: ℹ️ Processing your query...
Status: ℹ️ Query routed to ORBITQA agent
Agent: 🔵 OrbitQA

[Agent]: 🔔 What inclination angle would you like? (Press Enter for default 98.6°)

[You]: 98.6

Status: ⚙️ Processing...

[Agent]: 🤖 Here's the plot of your sun-synchronous orbit at 800 km altitude.
[Plot appears showing orbital path]
✅ Status: success
```

## 🔒 Security Notes

- The UI runs locally on your machine
- WebSocket connection is not encrypted by default
- Do not expose the backend server to public networks without proper security
- Session IDs are unique per browser session

## 📚 Additional Resources

- **Backend API**: Check `/health` endpoint for backend status
- **WebSocket Protocol**: Review `websocket_server.py` for message formats
- **Agent Documentation**: See README.md for agent capabilities
- **Orbital Tools**: Check `app/core/orbital_tools.py` for available functions

## 🆘 Getting Help

If you encounter issues:

1. **Check Logs**: Look at terminal output for both UI and backend
2. **Review Error Messages**: Read error messages carefully
3. **Session ID**: Note your session ID for debugging
4. **Backend Health**: Test backend with: `curl http://localhost:8000/health`

## 🎉 Enjoy AstroBrain!

You're now ready to use the AstroBrain UI! Start asking questions about orbital mechanics, mission planning, and aerospace engineering. The AI agents are here to help you with calculations, visualizations, and comprehensive mission analysis.

**Happy Exploring! 🚀✨**
