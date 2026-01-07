# 🏠 AI Rug Visualizer - Production Edition

RugUSA-quality rug visualization with **real AI models** - no heuristics, no fallbacks.

## ✨ Features

- **MiDaS Depth Estimation** - Intel's state-of-the-art monocular depth
- **SAM Segmentation** - Facebook's Segment Anything Model for floor/furniture
- **Depth-Aware Placement** - Rugs scale realistically with perspective
- **Furniture Occlusion** - Objects properly layer over rugs
- **Real-Time Manipulation** - Drag, scale, rotate with live preview
- **Production Quality** - Modular, type-safe, enterprise-ready code

---

## 🚀 Quick Start

### Option 1: Local Setup (Recommended for GPU)

**Requirements:**
- Python 3.9+
- Node.js 18+
- CUDA GPU (optional but recommended)
- 8GB RAM minimum, 16GB recommended

**1. Clone/Setup:**
```bash
# Create project directory
mkdir rug-visualizer && cd rug-visualizer

# Create backend structure
mkdir -p backend/{ai,core,models,uploads,outputs,cache}
mkdir -p frontend/src/{components,services,utils}
```

**2. Backend Setup:**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (2.4GB SAM model)
python download_models.py

# Start backend
python main.py
```

The backend will:
- Download MiDaS automatically on first run (via torch.hub)
- Preload models (takes ~30 seconds)
- Start API on http://localhost:8000

**3. Frontend Setup:**
```bash
cd ../frontend

# Initialize Vite project (if not exists)
npm create vite@latest . -- --template react

# Install dependencies
npm install lucide-react

# Start frontend
npm run dev
```

Frontend runs on http://localhost:5173

---

### Option 2: Kaggle (GPU Available)

**Why Kaggle?**
- Free GPU (P100)
- Pre-installed PyTorch
- 16GB RAM
- Perfect for AI models

**Setup:**

1. **Create New Notebook:**
   - Go to kaggle.com
   - Click "New Notebook"
   - Enable GPU: Settings → Accelerator → GPU P100

2. **Install Dependencies:**
```python
# Cell 1: Install packages
!pip install -q fastapi uvicorn python-multipart timm
!pip install -q git+https://github.com/facebookresearch/segment-anything.git
```

3. **Upload Backend Files:**
```python
# Cell 2: Create structure
!mkdir -p backend/{ai,core,models,uploads,outputs,cache}
```

Upload all Python files from artifact to respective folders.

4. **Download Models:**
```python
# Cell 3: Download SAM
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
      -O backend/models/sam_vit_h_4b8939.pth
```

5. **Run Backend:**
```python
# Cell 4: Start API
import subprocess
import time

# Start backend in background
proc = subprocess.Popen(['python', 'backend/main.py'])
time.sleep(10)  # Wait for startup

print("Backend running on port 8000")
```

6. **Access API:**
   - Kaggle provides public URLs
   - Use ngrok or Kaggle's URL forwarding
   - Update frontend API_BASE to Kaggle URL

---

## 📁 Project Structure

```
rug-visualizer/
├── backend/
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── depth_estimator.py      # MiDaS depth
│   │   └── sam_segmenter.py        # SAM segmentation
│   ├── core/
│   │   ├── __init__.py
│   │   └── compositor.py           # Rug compositing
│   ├── models/
│   │   └── sam_vit_h_4b8939.pth   # Downloaded
│   ├── uploads/                    # User uploads
│   ├── outputs/                    # Generated images
│   ├── cache/                      # Analysis cache
│   ├── config.py                   # Configuration
│   ├── main.py                     # FastAPI app
│   ├── download_models.py          # Model downloader
│   └── requirements.txt            # Python deps
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── RoomUpload.jsx      # Upload component
    │   │   ├── CanvasDisplay.jsx   # Canvas renderer
    │   │   ├── RugCatalog.jsx      # Rug selector
    │   │   └── RugControls.jsx     # Adjustment controls
    │   ├── services/
    │   │   └── api.js              # API client
    │   ├── utils/
    │   │   └── rugGenerator.js     # Pattern generator
    │   ├── App.jsx                 # Main app
    │   └── main.jsx                # Entry point
    ├── package.json
    └── vite.config.js
```

---

## 🔧 Configuration

**Backend** (`backend/config.py`):
```python
MIDAS_MODEL = "DPT_Large"     # Best quality
SAM_MODEL = "vit_h"           # ViT-H (2.4GB)
MAX_IMAGE_SIZE = 2048         # Resize large images
DEVICE = "cuda"               # or "cpu"
```

**Frontend** (`.env`):
```env
VITE_API_URL=http://localhost:8000
```

---

## 🎯 Usage

1. **Upload Room Photo**
   - Click upload area
   - Select clear, well-lit room photo
   - Wait for AI analysis (~10-20 seconds)

2. **Select Rug**
   - Choose from catalog
   - Rug appears on floor automatically

3. **Adjust Placement**
   - Drag rug to reposition
   - Scale: Slider to resize
   - Rotate: Slider to rotate
   - Depth Scaling: Toggle for perspective
   - Occlusion: Toggle for furniture layering

4. **Export**
   - Click "Export HD"
   - Saves final composited image

---

## 🐛 Troubleshooting

### Backend Issues

**"Failed to load MiDaS model"**
```bash
pip install timm
# MiDaS requires timm for transformers
```

**"SAM checkpoint not found"**
```bash
python download_models.py
# Or manually download from GitHub
```

**"CUDA out of memory"**
```python
# In config.py:
DEVICE = "cpu"
MAX_IMAGE_SIZE = 1024  # Reduce size
```

**Backend not starting:**
```bash
# Check port 8000 is free
lsof -ti:8000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :8000   # Windows

# Check logs
python main.py
# Look for error messages
```

### Frontend Issues

**"Network Error"**
- Ensure backend is running (http://localhost:8000)
- Check CORS settings in backend
- Verify API_BASE in api.js

**"Rug not appearing"**
- Check browser console for errors
- Ensure analysis completed (green badge)
- Try refreshing page

---

## 🎨 Adding Custom Rugs

Edit `frontend/src/utils/rugGenerator.js`:

```javascript
export const RUG_PATTERNS = [
  // ... existing patterns
  {
    id: 7,
    name: 'Your Pattern',
    pattern: 'custom',
    colors: ['#HEX1', '#HEX2', '#HEX3']
  }
];

// Add pattern generator:
case 'custom':
  // Your drawing code
  ctx.fillStyle = colors[0];
  // ...
  break;
```

---

## 🏗️ Architecture

### AI Pipeline

```
Room Image
    ↓
1. MiDaS Depth Estimation
    ├─→ Depth Map (H×W float32)
    ↓
2. SAM Floor Segmentation
    ├─→ Floor Mask (H×W bool)
    ├─→ Floor Corners (4-8 points)
    ↓
3. SAM Furniture Segmentation
    ├─→ Furniture Masks (List[H×W bool])
    ↓
4. Cache Results
    └─→ .npz file
```

### Compositing Pipeline

```
Rug + Parameters
    ↓
1. Depth-Based Scaling
    ├─→ Sample depth at position
    ├─→ Calculate scale factor
    ↓
2. Transform Rug
    ├─→ Scale
    ├─→ Rotate
    ├─→ Perspective Warp
    ↓
3. Generate Shadow
    ├─→ From rug alpha
    ├─→ Blur based on depth
    ↓
4. Composite Layers
    ├─→ Room (base)
    ├─→ Shadow (on floor)
    ├─→ Rug (on floor)
    ├─→ Furniture (occlusion)
    ↓
Final Image
```

---

## 📊 Performance

**First Run** (cold start):
- Model loading: ~30s
- MiDaS download: ~500MB (automatic)
- SAM checkpoint: ~2.4GB (manual)

**Subsequent Runs:**
- Upload + Analysis: 10-20s
- Compositing: 0.3-1s (real-time)
- Export: <0.1s

**GPU vs CPU:**
- GPU (CUDA): 10x faster analysis
- CPU: Acceptable for single use
- Kaggle GPU: Free & fast

---

## 🚢 Deployment

### Docker (Production)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
RUN python download_models.py

EXPOSE 8000
CMD ["python", "main.py"]
```

### Cloud Options

- **AWS EC2**: g4dn.xlarge (GPU)
- **Google Cloud**: n1-standard-4 + T4 GPU
- **Kaggle**: Free notebooks with P100
- **Vercel**: Frontend only (separate backend)

---

## 📝 API Documentation

### POST `/upload/room`
Upload room image
- **Body**: `multipart/form-data` with `file`
- **Returns**: `{image_id, width, height}`

### POST `/analyze/complete`
Analyze room with AI
- **Body**: `form-data` with `image_id`
- **Returns**: `{floor_mask_base64, depth_map_base64, floor_confidence, furniture_count}`
- **Time**: 10-20s

### POST `/composite/realtime`
Composite rug in real-time
- **Body**: JSON with `{room_image_id, rug_data_url, position_x, position_y, base_scale, rotation, use_depth, use_furniture_occlusion}`
- **Returns**: `{image_base64}`
- **Time**: 0.3-1s

---

## 🤝 Contributing

This is a production-ready template. To improve:

1. **Better Models**: Try Depth-Anything-v2 instead of MiDaS
2. **More Rugs**: Add real product images
3. **Advanced Features**: Lighting adjustment, texture mapping
4. **Mobile**: Touch controls, responsive design

---

## 📜 License

MIT - Use freely for commercial projects

---

## 🆘 Support

**Issues?**
1. Check troubleshooting section
2. Enable debug logging: `logging.basicConfig(level=logging.DEBUG)`
3. Check model files exist in `backend/models/`
4. Verify GPU available: `torch.cuda.is_available()`

**Contact:** Open GitHub issue or discussion

---

Made with ❤️ for production AI applications