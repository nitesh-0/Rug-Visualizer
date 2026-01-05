# 🏠 Rug Visualizer - AI-Powered Room Decorator

Complete setup guide for the integrated frontend + backend application.

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (for backend)
- **Node.js 18+** (for frontend)
- **8GB RAM** minimum
- **Optional:** NVIDIA GPU for faster AI

### Installation

#### 1. Clone/Setup Project
```bash
cd rug-visualizer
```

#### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download AI models (optional but recommended)
python download_models.py
```

#### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install
```

### Running the Application

#### Terminal 1: Start Backend
```bash
cd backend
python main.py
```
Backend will run on: `http://localhost:8000`

#### Terminal 2: Start Frontend
```bash
cd frontend
npm run dev
```
Frontend will run on: `http://localhost:3000`

#### Open Browser
Navigate to: `http://localhost:3000`

## 🎯 Features

- ✅ AI-powered floor detection
- ✅ Automatic furniture segmentation
- ✅ Realistic perspective matching
- ✅ Interactive rug manipulation
- ✅ Real-time preview
- ✅ High-resolution export

## 📖 Usage

1. Upload a room photo
2. Wait for AI to detect floor (~2-3 seconds)
3. Select a rug from catalog
4. Drag and adjust using sliders
5. Export your visualization

## 🛠️ Tech Stack

**Frontend:** React, Vite, TailwindCSS, Axios
**Backend:** FastAPI, PyTorch, OpenCV, Pillow
**AI Models:** Segment Anything (SAM), Custom segmentation

## 🐛 Troubleshooting

**Backend won't start:**
```bash
# Check if port 8000 is free
# Windows:
netstat -ano | findstr :8000
# Mac/Linux:
lsof -i :8000
```

**Frontend can't connect to backend:**
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify CORS settings in main.py

**AI models not working:**
- Models will fallback to heuristics automatically
- Download models using `python download_models.py`
- Check `backend/models/` directory

## 📝 License

MIT License