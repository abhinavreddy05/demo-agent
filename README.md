# Ahoum Voice Agent Example

The example includes a custom Next.js frontend and Python agent.

## Running the example

### Prerequisites

- Node.js
- Python 3.9-3.12
- LiveKit Cloud account (or OSS LiveKit server)
- Groq API key

### Frontend

Copy `.env.example` to `.env.local` and set the environment variables. Then run:

```bash
cd frontend
npm install
npm run dev
```

### Agent

Copy `.env.example` to `.env` and set the environment variables. Then run:

```bash
cd agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py dev
```
