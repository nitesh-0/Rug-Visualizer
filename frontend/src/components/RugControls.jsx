import React from 'react';
import { Move, RotateCcw } from 'lucide-react';

export default function RugControls({
  selectedRug,
  rugScale,
  rugRotation,
  useDepth,
  useFurnitureOcclusion,
  onScaleChange,
  onRotationChange,
  onDepthToggle,
  onOcclusionToggle,
  onReset
}) {
  if (!selectedRug) return null;

  return (
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
            onChange={(e) => onScaleChange(parseFloat(e.target.value))} 
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
            onChange={(e) => onRotationChange(parseFloat(e.target.value))} 
            className="w-full h-2 bg-stone-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
          />
        </div>
        
        <div className="flex items-end">
          <button
            onClick={onReset}
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
            onChange={(e) => onDepthToggle(e.target.checked)}
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
            onChange={(e) => onOcclusionToggle(e.target.checked)}
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
  );
}