import React from 'react';
import { Upload } from 'lucide-react';

export default function RoomUpload({ onUpload, isUploading }) {
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <label className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer z-10 bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-dashed border-amber-300 rounded-2xl m-3 hover:border-amber-500 hover:bg-amber-100/50 transition-all">
      <input 
        type="file" 
        accept="image/*" 
        onChange={handleFileChange} 
        className="hidden"
        disabled={isUploading}
      />
      
      <div className="w-20 h-20 bg-gradient-to-br from-amber-500 to-orange-500 rounded-3xl flex items-center justify-center mb-5 shadow-xl">
        <Upload className="w-10 h-10 text-white" />
      </div>
      
      <h3 className="font-bold text-stone-800 text-xl mb-2">
        {isUploading ? 'Uploading...' : 'Upload Room Photo'}
      </h3>
      
      <p className="text-sm text-stone-600 text-center max-w-md mb-3">
        AI will analyze depth, detect floor, and identify furniture for realistic rug placement
      </p>
      
      <div className="flex gap-2 text-xs text-stone-500">
        <span className="px-2 py-1 bg-white rounded-full">✓ Depth Estimation</span>
        <span className="px-2 py-1 bg-white rounded-full">✓ Floor Detection</span>
        <span className="px-2 py-1 bg-white rounded-full">✓ Furniture Occlusion</span>
      </div>
    </label>
  );
}