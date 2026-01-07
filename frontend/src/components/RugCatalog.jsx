import React from 'react';
import { Layers } from 'lucide-react';

export default function RugCatalog({ rugs, rugPreviews, selectedRug, onSelectRug }) {
  return (
    <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-stone-200">
      <div className="p-4 bg-gradient-to-r from-amber-50 to-orange-50 border-b border-stone-200">
        <h3 className="font-bold text-stone-800 flex items-center gap-2">
          <Layers className="w-5 h-5 text-amber-600" />
          Rug Catalog
        </h3>
        <p className="text-xs text-stone-600 mt-1">Select a rug to place in your room</p>
      </div>
      
      <div className="p-3 grid grid-cols-2 gap-3 max-h-96 overflow-y-auto">
        {rugs.map((rug) => (
          <div
            key={rug.id}
            onClick={() => onSelectRug(rug)}
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
  );
}