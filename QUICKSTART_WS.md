# AstroBrain WebSocket Quick Start Guide

## Overview

AstroBrain WebSocket server enables real-time, bidirectional communication between clients and the AstroBrain AI agents (OrbitQA and MissionOps). The server handles agent interrupts for clarifications, allowing users to provide additional information when the agent needs it.

## Starting the Server

### 1. Activate Virtual Environment

```bash
# Windows
.\myenv\Scripts\activate

# Linux/Mac
source myenv/bin/activate
```

### 2. Start the WebSocket Server

```bash
python -m app.websocket_server
```

By default, the server starts on `ws://localhost:8000`

### 3. Custom Host/Port (Optional)

Set environment variables in your `.env` file:

```env
WS_HOST=0.0.0.0
WS_PORT=8000
```

Or run with custom settings:

```bash
# In Python
from app.websocket_server import start_server
start_server(host="127.0.0.1", port=8080)
```

## WebSocket Endpoint

```
ws://localhost:8000/ws/{thread_id}
```

- **thread_id**: A unique identifier for each user session (use UUID v4)

## Message Protocol

All messages are JSON-formatted.

### 1. Client → Server Messages

#### Send a Query

```json
{
  "type": "query",
  "message": "Calculate orbital period for LEO at 400km altitude"
}
```

**Note**: The system automatically selects the appropriate agent (OrbitQA or MissionOps) based on your query content using the AstroBrain router.

#### Send Clarification Response

When the agent requests clarification, respond with:

```json
{
  "type": "clarification",
  "message": "The satellite mass is 500 kg"
}
```

#### Ping (Keep-Alive)

```json
{
  "type": "ping"
}
```

### 2. Server → Client Messages

#### Status Update

```json
{
  "type": "status",
  "message": "Processing your query...",
  "status": "processing"
}
```

#### Agent Selection Notification

```json
{
  "type": "status",
  "message": "Query routed to ORBITQA agent",
  "agent": "orbitqa"
}
```

#### Clarification Request (Interrupt)

When the agent needs more information:

```json
{
  "type": "clarification_request",
  "message": "What is the satellite mass in kilograms?",
  "timestamp": 1234567890.123
}
```

**Action Required**: Client must send a clarification response message.

#### Final Response

```json
{
  "type": "response",
  "data": {
    "isInterrupted": false,
    "clarification_limit_exceeded": false,
    "interrupt_message": "",
    "final_response": {
      "status": "success",
      "reason": "...",
      "message": "...",
      "plots": [],
      "warnings": []
    }
  },
  "timestamp": 1234567890.456
}
```

#### Error Response

```json
{
  "type": "error",
  "message": "Error description",
  "code": "ERROR_CODE"
}
```

Error codes:
- `TIMEOUT`: User did not respond to clarification in time
- `PROCESSING_ERROR`: Error during query processing

#### Pong Response

```json
{
  "type": "pong"
}
```

## Example Client (Python)

```python
import asyncio
import json
import uuid
import websockets

async def astrobrain_client():
    thread_id = str(uuid.uuid4())
    uri = f"ws://localhost:8000/ws/{thread_id}"
    
    async with websockets.connect(uri) as websocket:
        # Send query (agent auto-selected)
        query = {
            "type": "query",
            "message": "What is the orbital period at 400km altitude?"
        }
        await websocket.send(json.dumps(query))
        print(f"Sent: {query}")
        
        # Listen for responses
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Received: {data}")
            
            # Handle different message types
            if data["type"] == "clarification_request":
                # Agent needs clarification
                print(f"Agent asks: {data['message']}")
                user_input = input("Your answer: ")
                
                clarification = {
                    "type": "clarification",
                    "message": user_input
                }
                await websocket.send(json.dumps(clarification))
                
            elif data["type"] == "response":
                # Final answer received
                print(f"Final Response: {data['data']}")
                break
                
            elif data["type"] == "error":
                # Error occurred
                print(f"Error: {data['message']}")
                break

# Run the client
asyncio.run(astrobrain_client())
```

## Example Client (JavaScript/Browser)

```javascript
const threadId = crypto.randomUUID();
const ws = new WebSocket(`ws://localhost:8000/ws/${threadId}`);

ws.onopen = () => {
    console.log('Connected to AstroBrain');
    
    // Send query (agent auto-selected by router)
    ws.send(JSON.stringify({
        type: 'query',
        message: 'Calculate the orbital period at 400km altitude'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    switch(data.type) {
        case 'status':
            console.log(`Status: ${data.message}`);
            break;
            
        case 'clarification_request':
            // Agent needs clarification
            const answer = prompt(data.message);
            ws.send(JSON.stringify({
                type: 'clarification',
                message: answer
            }));
            break;
            
        case 'response':
            // Final answer
            console.log('Final Response:', data.data);
            ws.close();
            break;
            
        case 'error':
            console.error('Error:', data.message);
            ws.close();
            break;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from AstroBrain');
};
```

## Clarification Flow Example

1. **Client sends query**:
   ```json
   {"type": "query", "message": "Assess mission feasibility for LEO satellite"}
   ```

2. **Server sends status**:
   ```json
   {"type": "status", "message": "Processing your query...", "status": "processing"}
   ```

3. **Server notifies agent selection**:
   ```json
   {"type": "status", "message": "Query routed to MISSIONOPS agent", "agent": "missionops"}
   ```

3. **Agent needs clarification** (interrupt):
   ```json
   {
     "type": "clarification_request",
     "message": "What is the satellite mass in kilograms?"
   }
   ```

4. **Client provides clarification**:
   ```json
   {"type": "clarification", "message": "500 kg"}
   ```

5. **Agent may ask more questions** (up to 5 clarifications):
   ```json
   {
     "type": "clarification_request",
     "message": "What is the orbital altitude in kilometers?"
   }
   ```

6. **Client responds again**:
   ```json
   {"type": "clarification", "message": "400 km"}
   ```

7. **Server sends final response**:
   ```json
   {
     "type": "response",
     "data": {
       "isInterrupted": false,
       "final_response": {
         "status": "success",
         "message": "Mission is feasible...",
         "warnings": []
       }
     }
   }
   ```

## Important Notes

### Timeouts

- **Clarification timeout**: 600 seconds (10 minutes)
- **Clarification limit**: Maximum 5 clarifications per query
- If timeout or limit is exceeded, the session terminates with an error

### Thread IDs

- Each client connection must use a unique `thread_id` (recommend UUID v4)
- Thread ID persists for the duration of the query processing
- Use the same thread ID for the entire conversation including clarifications

### Agent Selection

- Agent selection is **fully automatic** - the system uses the AstroBrain router to analyze your query
- **OrbitQA**: Automatically selected for simple orbital calculations, plotting, and conceptual questions
- **MissionOps**: Automatically selected for mission feasibility, multi-stage analysis, and resource constraints
- You'll receive a status message indicating which agent was selected for your query

### Connection Management

- WebSocket connections are cleaned up automatically on disconnect
- Pending tasks are cancelled when connection drops
- Each connection is independent with its own processing context

## Health Check Endpoints

### Root Endpoint
```
GET http://localhost:8000/
```

Returns server information and available endpoints.

### Health Check
```
GET http://localhost:8000/health
```

Returns:
```json
{
  "status": "healthy",
  "active_connections": 2,
  "processing_tasks": 1
}
```

## Troubleshooting

### Connection Refused

Ensure the server is running:
```bash
python -m app.websocket_server
```

### Timeout Errors

- Check if client is responding to clarification requests within 10 minutes
- Ensure proper JSON formatting in messages
- Verify network connectivity

### Agent Selection Issues

- Review query wording - be specific about requirements
- Manually specify agent using `"agent": "orbitqa"` or `"agent": "missionops"`

### Environment Issues

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

And `.env` file contains necessary configuration:
```env
# Model configurations
model_name_astrobrain=llama3.1
model_temperature_astrobrain=0.0

# Add other model configurations for orbitqa and missionops nodes
# (refer to existing .env setup)
```

## Production Deployment

For production deployment:

1. **Use a process manager** (e.g., systemd, supervisor, PM2)
2. **Enable SSL/TLS** for secure WebSocket connections (wss://)
3. **Set up reverse proxy** (nginx, Apache) for load balancing
4. **Configure CORS** if serving web clients from different origins
5. **Monitor logs** for errors and performance issues
6. **Set resource limits** to prevent memory/CPU exhaustion

Example with gunicorn + uvicorn:
```bash
gunicorn app.websocket_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Support

For issues or questions, refer to the main project documentation or contact the development team.
