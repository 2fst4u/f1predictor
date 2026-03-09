python main.py --web --port 8000 &
PID=$!
sleep 2
curl -s http://localhost:8000 | grep -q "F1 OUTCOME PREDICTOR" && echo "UI loaded successfully"
kill $PID
