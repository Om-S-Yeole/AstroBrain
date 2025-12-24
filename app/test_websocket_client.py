"""
Simple WebSocket test client for AstroBrain
Tests the query and clarification flow
"""

import asyncio
import json
import uuid

import websockets


async def test_client():
    """Test the AstroBrain WebSocket server with a sample query."""
    thread_id = str(uuid.uuid4())
    uri = f"ws://localhost:8000/ws/{thread_id}"

    print(f"Connecting to {uri}...")
    print(f"Thread ID: {thread_id}")
    print("-" * 60)

    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")

            # Send a test query
            query_message = {
                "type": "query",
                "message": "What is the orbital period for a satellite at 400 km altitude?",
            }

            print(f"\n📤 Sending query: {query_message['message']}")
            await websocket.send(json.dumps(query_message))

            # Listen for responses
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)

                    print(f"\n📥 Received message type: {data['type']}")

                    if data["type"] == "status":
                        print(f"   Status: {data['message']}")
                        if "agent" in data:
                            print(f"   Agent: {data['agent']}")

                    elif data["type"] == "clarification_request":
                        # Agent needs clarification
                        print(f"\n❓ Agent asks: {data['message']}")
                        user_input = input("   Your answer: ")

                        clarification = {"type": "clarification", "message": user_input}
                        await websocket.send(json.dumps(clarification))
                        print("   ✓ Clarification sent")

                    elif data["type"] == "response":
                        # Final answer received
                        print("\n✅ Final Response:")
                        print(json.dumps(data["data"], indent=2))
                        break

                    elif data["type"] == "error":
                        # Error occurred
                        print(f"\n❌ Error: {data['message']}")
                        if "code" in data:
                            print(f"   Code: {data['code']}")
                        break

                    elif data["type"] == "pong":
                        print("   Pong received (keep-alive)")

                except websockets.exceptions.ConnectionClosed:
                    print("\n⚠ Connection closed by server")
                    break

            print("\n" + "-" * 60)
            print("Test completed!")

    except ConnectionRefusedError:
        print("❌ Connection refused. Is the WebSocket server running?")
        print("   Start it with: python -m app.websocket_server")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("AstroBrain WebSocket Test Client")
    print("=" * 60)

    asyncio.run(test_client())
