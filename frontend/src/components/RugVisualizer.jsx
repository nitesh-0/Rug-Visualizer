import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, RotateCcw, ZoomIn, ZoomOut, Download, Move, Maximize2, Layers, AlertCircle, CheckCircle, Loader, Eye, EyeOff } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

// Sample rug patterns
const RUG_PATTERNS = [
  { id: 1, name: 'Modern Grid', pattern: 'grid', colors: ['#E8DFD5', '#C9A962', '#8B7355'] },
  { id: 2, name: 'Persian Classic', pattern: 'persian', colors: ['#8B2323', '#C9A962', '#2D2D2D'] },
  { id: 3, name: 'Boho Diamond', pattern: 'bohemian', colors: ['#C67B5C', '#8FA68A', '#C9A962'] },
  { id: 4, name: 'Minimal Stripe', pattern: 'stripe', colors: ['#F5F1EB', '#C67B5C', '#E8DFD5'] },
  { id: 5, name: 'Geometric', pattern: 'geometric', colors: ['#C9A962', '#E8DFD5', '#8B7355'] },
  { id: 6, name: 'Solid Sage', pattern: 'solid', colors: ['#8FA68A'] },
];

// Generate rug pattern as data URL
const generateRugPattern = (pattern, colors, width = 400, height = 280) => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  ctx.fillStyle = colors[0] || '#E8DFD5';
  ctx.fillRect(0, 0, width, height);
  
  switch (pattern) {
    case 'grid':
      ctx.fillStyle = colors[1] || '#C9A962';
      for (let x = 0; x < width; x += 50) ctx.fillRect(x, 0, 20, height);
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = colors[2] || '#8B7355';
      for (let y = 0; y < height; y += 50) ctx.fillRect(0, y, width, 20);
      ctx.globalAlpha = 1;
      break;
    case 'persian':
      ctx.strokeStyle = colors[1] || '#C9A962';
      ctx.lineWidth = 12;
      ctx.strokeRect(12, 12, width - 24, height - 24);
      ctx.beginPath();
      ctx.ellipse(width/2, height/2, 80, 55, 0, 0, Math.PI * 2);
      ctx.fillStyle = colors[1];
      ctx.fill();
      ctx.strokeStyle = colors[2] || '#2D2D2D';
      ctx.lineWidth = 3;
      ctx.stroke();
      break;
    case 'bohemian':
      const size = 40;
      for (let y = -size; y < height + size; y += size) {
        for (let x = -size; x < width + size; x += size) {
          const offset = ((y / size) % 2) * (size / 2);
          ctx.beginPath();
          ctx.moveTo(x + offset + size/2, y);
          ctx.lineTo(x + offset + size, y + size/2);
          ctx.lineTo(x + offset + size/2, y + size);
          ctx.lineTo(x + offset, y + size/2);
          ctx.closePath();
          ctx.fillStyle = colors[Math.floor((x + y) / size) % colors.length];
          ctx.fill();
        }
      }
      break;
    case 'stripe':
      for (let y = 0; y < height; y += 20) {
        ctx.fillStyle = colors[(y / 20) % colors.length];
        ctx.fillRect(0, y, width, 20);
      }
      break;
    case 'geometric':
      const gridSize = 35;
      for (let y = 0; y < height; y += gridSize) {
        for (let x = 0; x < width; x += gridSize) {
          ctx.fillStyle = colors[((x / gridSize) + (y / gridSize)) % colors.length];
          ctx.fillRect(x + 3, y + 3, gridSize - 6, gridSize - 6);
        }
      }
      break;
    case 'solid':
      ctx.fillStyle = 'rgba(0,0,0,0.03)';
      for (let i = 0; i < 500; i++) {
        ctx.fillRect(Math.random() * width, Math.random() * height, 2, 2);
      }
      break;
  }
  
  ctx.strokeStyle = 'rgba(0,0,0,0.15)';
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, width, height);
  
  return canvas.toDataURL();
};

export default function EnhancedRugVisualizer() {
  const canvasRef = useRef(null);
  const [roomImage, setRoomImage] = useState(null);
  const [roomImageId, setRoomImageId] = useState(null);
  const [selectedRug, setSelectedRug] = useState(null);
  const [rugPosition, setRugPosition] = useState({ x: 0.5, y: 0.6 });
  const [rugScale, setRugScale] = useState(1.0);
  const [rugRotation, setRugRotation] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  
  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [floorConfidence, setFloorConfidence] = useState(0);
  const [furnitureCount, setFurnitureCount] = useState(0);
  const [depthMap, setDepthMap] = useState(null);
  
  // Visualization state
  const [compositedImage, setCompositedImage] = useState(null);
  const [isCompositing, setIsCompositing] = useState(false);
  const [showDepthOverlay, setShowDepthOverlay] = useState(false);
  const [useDepth, setUseDepth] = useState(true);
  const [useFurnitureOcclusion, setUseFurnitureOcclusion] = useState(true);
  
  const [error, setError] = useState(null);
  const [rugPreviews, setRugPreviews] = useState({});

  // Generate rug previews
  useEffect(() => {
    const previews = {};
    RUG_PATTERNS.forEach(rug => {
      previews[rug.id] = generateRugPattern(rug.pattern, rug.colors);
    });
    setRugPreviews(previews);
  }, []);

  // Auto-composite when parameters change
  useEffect(() => {
    if (roomImageId && selectedRug && analysisComplete) {
      const timeoutId = setTimeout(() => {
        compositeRug();
      }, 300); // Debounce
      return () => clearTimeout(timeoutId);
    }
  }, [roomImageId, selectedRug, rugPosition, rugScale, rugRotation, useDepth, useFurnitureOcclusion, analysisComplete]);

  // Handle room upload
  const handleRoomUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setError(null);
    setAnalysisComplete(false);
    setCompositedImage(null);
    
    try {
      // Display preview
      const reader = new FileReader();
      reader.onload = (event) => {
        setRoomImage(event.target.result);
      };
      reader.readAsDataURL(file);
      
      // Upload to backend
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadRes = await fetch(`${API_BASE}/upload/room`, {
        method: 'POST',
        body: formData
      });
      
      if (!uploadRes.ok) throw new Error('Upload failed');
      
      const uploadData = await uploadRes.json();
      setRoomImageId(uploadData.image_id);
      
      // Trigger complete analysis
      await analyzeRoom(uploadData.image_id);
      
    } catch (err) {
      console.error('Upload error:', err);
      setError('Failed to upload image. Make sure backend is running on port 8000.');
    }
  };

  // AI Analysis
  const analyzeRoom = async (imageId) => {
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('image_id', imageId);
      
      const response = await fetch(`${API_BASE}/analyze/complete`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) throw new Error('Analysis failed');
      
      const data = await response.json();
      
      if (data.success) {
        setFloorConfidence(data.floor_confidence);
        setFurnitureCount(data.furniture_count);
        setDepthMap(data.depth_map_base64);
        setAnalysisComplete(true);
      } else {
        throw new Error(data.message || 'Analysis failed');
      }
      
    } catch (err) {
      console.error('Analysis error:', err);
      setError('AI analysis failed. Using basic placement mode.');
      setAnalysisComplete(true); // Continue with fallback
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Composite rug
  const compositeRug = async () => {
    if (!roomImageId || !selectedRug) return;
    
    setIsCompositing(true);
    
    try {
      const response = await fetch(`${API_BASE}/composite/realtime`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          room_image_id: roomImageId,
          rug_data_url: rugPreviews[selectedRug.id],
          position_x: rugPosition.x,
          position_y: rugPosition.y,
          base_scale: rugScale,
          rotation: rugRotation,
          use_depth: useDepth,
          use_furniture_occlusion: useFurnitureOcclusion
        })
      });
      
      if (!response.ok) throw new Error('Compositing failed');
      
      const data = await response.json();
      
      if (data.success) {
        setCompositedImage(`data:image/png;base64,${data.image_base64}`);
      } else {
        throw new Error(data.message);
      }
      
    } catch (err) {
      console.error('Composite error:', err);
      setError('Compositing failed. Check console for details.');
    } finally {
      setIsCompositing(false);
    }
  };

  // Mouse handlers for dragging
  const handleMouseDown = (e) => {
    if (!selectedRug || !compositedImage) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    
    setIsDragging(true);
    setDragStart({ x: x - rugPosition.x, y: y - rugPosition.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    
    setRugPosition({
      x: Math.max(0.1, Math.min(0.9, x - dragStart.x)),
      y: Math.max(0.1, Math.min(0.9, y - dragStart.y))
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Export image
  const exportImage = () => {
    if (!compositedImage) return;
    
    const link = document.createElement('a');
    link.download = 'room-with-rug-enhanced.png';
    link.href = compositedImage;
    link.click();
  };

  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw composited result or room image
    const imgSrc = compositedImage || roomImage;
    
    if (imgSrc) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // Overlay depth map if enabled
        if (showDepthOverlay && depthMap) {
          const depthImg = new Image();
          depthImg.onload = () => {
            ctx.globalAlpha = 0.4;
            ctx.drawImage(depthImg, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
          };
          depthImg.src = `data:image/png;base64,${depthMap}`;
        }
      };
      img.src = imgSrc;
    } else {
      // Empty state
      ctx.fillStyle = '#F5F1EB';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#A0A0A0';
      ctx.font = '16px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('Upload a room photo to begin', canvas.width / 2, canvas.height / 2);
    }
  }, [compositedImage, roomImage, showDepthOverlay, depthMap]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-50 via-amber-50 to-orange-50">
      {/* Header */}
      <header className="bg-white/90 backdrop-blur-lg border-b border-stone-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-amber-600 via-orange-500 to-red-500 rounded-xl flex items-center justify-center shadow-lg">
              <Layers className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-stone-800 text-lg">AI Rug Visualizer Pro</h1>
              <p className="text-xs text-stone-500">Depth-Aware • Real-Time • Production Quality</p>
            </div>
          </div>
          
          <button 
            onClick={exportImage}
            disabled={!compositedImage}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-amber-600 to-orange-600 text-white rounded-xl text-sm font-semibold hover:from-amber-700 hover:to-orange-700 transition-all shadow-lg disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            Export HD
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Error Alert */}
        {error && (
          <div className="mb-4 bg-red-50 border-2 border-red-200 rounded-xl p-4 flex items-start gap-3 shadow-sm">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-red-900 text-sm">Error</h3>
              <p className="text-red-700 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Main Canvas */}
          <div className="lg:col-span-3 space-y-4">
            <div className="bg-white rounded-2xl shadow-2xl overflow-hidden relative border border-stone-200">
              {/* Upload overlay */}
              {!roomImage && (
                <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer z-10 bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-dashed border-amber-300 rounded-2xl m-3 hover:border-amber-500 hover:bg-amber-100/50 transition-all">
                  <input type="file" accept="image/*" onChange={handleRoomUpload} className="hidden" />
                  <div className="w-20 h-20 bg-gradient-to-br from-amber-500 to-orange-500 rounded-3xl flex items-center justify-center mb-5 shadow-xl">
                    <Upload className="w-10 h-10 text-white" />
                  </div>
                  <h3 className="font-bold text-stone-800 text-xl mb-2">Upload Room Photo</h3>
                  <p className="text-sm text-stone-600 text-center max-w-md mb-3">
                    Our AI will analyze depth, detect floor, and identify furniture for realistic rug placement
                  </p>
                  <div className="flex gap-2 text-xs text-stone-500">
                    <span className="px-2 py-1 bg-white rounded-full">✓ Depth Estimation</span>
                    <span className="px-2 py-1 bg-white rounded-full">✓ Floor Detection</span>
                    <span className="px-2 py-1 bg-white rounded-full">✓ Furniture Occlusion</span>
                  </div>
                </label>
              )}
              
              {/* Canvas */}
              <canvas
                ref={canvasRef}
                width={1000}
                height={600}
                className="w-full h-auto cursor-move"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
              
              {/* Analysis overlay */}
              {isAnalyzing && (
                <div className="absolute inset-0 bg-white/95 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                  <div className="w-16 h-16 border-4 border-amber-200 border-t-amber-600 rounded-full animate-spin mb-5" />
                  <h3 className="text-stone-800 font-bold text-lg mb-2">AI Analysis in Progress</h3>
                  <div className="space-y-2 text-sm text-stone-600">
                    <p className="flex items-center gap-2">
                      <Loader className="w-4 h-4 animate-spin" />
                      Running depth estimation (MiDaS)...
                    </p>
                    <p className="flex items-center gap-2">
                      <Loader className="w-4 h-4 animate-spin" />
                      Detecting floor region...
                    </p>
                    <p className="flex items-center gap-2">
                      <Loader className="w-4 h-4 animate-spin" />
                      Segmenting furniture...
                    </p>
                  </div>
                </div>
              )}
              
              {/* Analysis complete badge */}
              {analysisComplete && !isAnalyzing && (
                <div className="absolute top-4 left-4 space-y-2">
                  <div className="bg-emerald-500/95 backdrop-blur text-white px-4 py-2 rounded-xl text-sm font-semibold flex items-center gap-2 shadow-lg">
                    <CheckCircle className="w-4 h-4" />
                    AI Analysis Complete
                  </div>
                  <div className="bg-white/95 backdrop-blur px-3 py-1.5 rounded-lg text-xs text-stone-700 shadow-lg">
                    Floor: {Math.round(floorConfidence * 100)}% • Furniture: {furnitureCount} objects
                  </div>
                </div>
              )}
              
              {/* Compositing indicator */}
              {isCompositing && (
                <div className="absolute top-4 right-4 bg-blue-500/95 backdrop-blur text-white px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2 shadow-lg">
                  <Loader className="w-3 h-3 animate-spin" />
                  Rendering...
                </div>
              )}
              
              {/* Canvas controls */}
              <div className="absolute bottom-4 right-4 flex gap-2">
                <button 
                  onClick={() => setShowDepthOverlay(!showDepthOverlay)}
                  disabled={!depthMap}
                  className="w-10 h-10 bg-white/95 backdrop-blur rounded-xl flex items-center justify-center text-stone-700 shadow-lg hover:bg-white transition-all disabled:opacity-40"
                  title={showDepthOverlay ? "Hide depth overlay" : "Show depth overlay"}
                >
                  {showDepthOverlay ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              
              {/* Change photo */}
              {roomImage && (
                <label className="absolute top-4 right-4 cursor-pointer">
                  <input type="file" accept="image/*" onChange={handleRoomUpload} className="hidden" />
                  <div className="flex items-center gap-2 px-4 py-2 bg-white/95 backdrop-blur rounded-xl text-stone-700 text-sm font-medium shadow-lg hover:bg-white transition-all">
                    <Upload className="w-4 h-4" />
                    Change Photo
                  </div>
                </label>
              )}
            </div>
            
            {/* Controls */}
            {selectedRug && analysisComplete && (
              <div className="bg-white rounded-2xl p-5 shadow-xl border border-stone-200">
                <h3 className="font-bold text-stone-800 mb-4 flex items-center gap-2">
                  <Move className="w-5 h-5 text-amber-600" />
                  Adjust {selectedRug.name}
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-5">
                  <div>
                    <label className="text-sm font-semibold text-stone-600 mb-2 block">
                      Size: {rugScale.toFixed(2)}x
                    </label>
                    <input 
                      type="range" 
                      min="0.4" 
                      max="2.5" 
                      step="0.1" 
                      value={rugScale} 
                      onChange={(e) => setRugScale(parseFloat(e.target.value))} 
                      className="w-full h-2 bg-stone-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                    />
                  </div>
                  
                  <div>
                    <label className="text-sm font-semibold text-stone-600 mb-2 block">
                      Rotation: {Math.round(rugRotation)}°
                    </label>
                    <input 
                      type="range" 
                      min="-180" 
                      max="180" 
                      step="5" 
                      value={rugRotation} 
                      onChange={(e) => setRugRotation(parseFloat(e.target.value))} 
                      className="w-full h-2 bg-stone-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
                    />
                  </div>
                  
                  <div className="flex items-end">
                    <button
                      onClick={() => {
                        setRugScale(1.0);
                        setRugRotation(0);
                        setRugPosition({ x: 0.5, y: 0.6 });
                      }}
                      className="w-full px-4 py-2 bg-stone-100 hover:bg-stone-200 text-stone-700 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
                    >
                      <RotateCcw className="w-4 h-4" />
                      Reset
                    </button>
                  </div>
                </div>
                
                {/* Advanced options */}
                <div className="border-t border-stone-200 pt-4 space-y-3">
                  <h4 className="text-sm font-semibold text-stone-700 mb-2">Advanced Options</h4>
                  
                  <label className="flex items-center gap-3 cursor-pointer group">
                    <input 
                      type="checkbox" 
                      checked={useDepth}
                      onChange={(e) => setUseDepth(e.target.checked)}
                      className="w-5 h-5 rounded border-stone-300 text-amber-600 focus:ring-amber-500"
                    />
                    <div>
                      <span className="text-sm font-medium text-stone-700 group-hover:text-stone-900">
                        Depth-Aware Scaling
                      </span>
                      <p className="text-xs text-stone-500">Rug size adjusts based on distance from camera</p>
                    </div>
                  </label>
                  
                  <label className="flex items-center gap-3 cursor-pointer group">
                    <input 
                      type="checkbox" 
                      checked={useFurnitureOcclusion}
                      onChange={(e) => setUseFurnitureOcclusion(e.target.checked)}
                      className="w-5 h-5 rounded border-stone-300 text-amber-600 focus:ring-amber-500"
                    />
                    <div>
                      <span className="text-sm font-medium text-stone-700 group-hover:text-stone-900">
                        Furniture Occlusion
                      </span>
                      <p className="text-xs text-stone-500">Furniture appears in front of rug (realistic layering)</p>
                    </div>
                  </label>
                </div>
                
                <p className="text-xs text-stone-500 mt-4 flex items-center gap-2 bg-amber-50 p-3 rounded-lg">
                  <Move className="w-4 h-4 text-amber-600" />
                  💡 Drag the rug directly on the canvas to reposition it
                </p>
              </div>
            )}
          </div>
          
          {/* Sidebar */}
          <div className="space-y-4">
            {/* Rug Catalog */}
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-stone-200">
              <div className="p-4 bg-gradient-to-r from-amber-50 to-orange-50 border-b border-stone-200">
                <h3 className="font-bold text-stone-800 flex items-center gap-2">
                  <Layers className="w-5 h-5 text-amber-600" />
                  Rug Catalog
                </h3>
                <p className="text-xs text-stone-600 mt-1">Select a rug to place in your room</p>
              </div>
              
              <div className="p-3 grid grid-cols-2 gap-3 max-h-96 overflow-y-auto">
                {RUG_PATTERNS.map((rug) => (
                  <div
                    key={rug.id}
                    onClick={() => setSelectedRug(rug)}
                    className={`rounded-xl overflow-hidden cursor-pointer transition-all transform hover:scale-105 border-2 ${
                      selectedRug?.id === rug.id
                        ? 'border-amber-500 shadow-xl shadow-amber-200 ring-2 ring-amber-200'
                        : 'border-stone-200 hover:border-amber-300 hover:shadow-lg'
                    }`}
                  >
                    <div className="aspect-[4/3] bg-stone-100">
                      {rugPreviews[rug.id] && (
                        <img src={rugPreviews[rug.id]} alt={rug.name} className="w-full h-full object-cover" />
                      )}
                    </div>
                    <div className="p-2 bg-white">
                      <p className="text-xs font-semibold text-stone-800 truncate">{rug.name}</p>
                      <div className="flex gap-1 mt-1.5">
                        {rug.colors.slice(0, 3).map((color, i) => (
                          <div 
                            key={i} 
                            className="w-4 h-4 rounded-full border-2 border-white shadow-sm" 
                            style={{ backgroundColor: color }} 
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* AI Features */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-5 border border-blue-200 shadow-lg">
              <h3 className="font-bold text-stone-800 mb-3 text-sm flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-blue-600" />
                AI-Powered Features
              </h3>
              <ul className="space-y-2">
                {[
                  'MiDaS depth estimation',
                  'Automatic floor detection',
                  'Furniture segmentation',
                  'Perspective-correct warping',
                  'Realistic shadows & lighting',
                  'Real-time rendering'
                ].map((feature, i) => (
                  <li key={i} className="flex items-center gap-2 text-xs text-stone-700">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" />
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
            
            {/* How to Use */}
            <div className="bg-white rounded-2xl p-5 shadow-xl border border-stone-200">
              <h3 className="font-bold text-stone-800 mb-3 text-sm">Quick Start Guide</h3>
              <ol className="space-y-3 text-xs text-stone-700">
                {[
                  { step: 'Upload your room photo', desc: 'Any photo with a visible floor' },
                  { step: 'Wait for AI analysis', desc: 'Takes 5-10 seconds' },
                  { step: 'Select a rug pattern', desc: 'From the catalog above' },
                  { step: 'Drag & adjust', desc: 'Position, scale, and rotate' },
                  { step: 'Export result', desc: 'Download high-quality image' }
                ].map((item, i) => (
                  <li key={i} className="flex gap-3">
                    <span className="w-6 h-6 bg-gradient-to-br from-amber-500 to-orange-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
                      {i + 1}
                    </span>
                    <div>
                      <p className="font-semibold text-stone-800">{item.step}</p>
                      <p className="text-stone-500">{item.desc}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
            
            {/* Tips */}
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-2xl p-4 border border-amber-200">
              <h3 className="font-bold text-stone-800 mb-2 text-sm">💡 Pro Tips</h3>
              <ul className="space-y-1.5 text-xs text-stone-700">
                <li>• Best results with clear, well-lit rooms</li>
                <li>• Include visible floor area in photo</li>
                <li>• Try depth overlay to see AI perception</li>
                <li>• Enable occlusion for realistic layering</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}