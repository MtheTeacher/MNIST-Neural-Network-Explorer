
import React from 'react';
import { XIcon } from '../constants';

interface InfoModalProps {
    title: string;
    onClose: () => void;
    children: React.ReactNode;
}

export const InfoModal: React.FC<InfoModalProps> = ({ title, onClose, children }) => {
    // Close on escape key press
    React.useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [onClose]);

    return (
        <div 
            className="fixed inset-0 bg-black/70 backdrop-blur-md flex justify-center items-center z-50 p-4 transition-opacity duration-300"
            onClick={onClose}
            role="dialog"
            aria-modal="true"
            aria-labelledby="info-modal-title"
        >
            <div 
                className="bg-gray-900 border border-white/20 rounded-2xl p-8 shadow-2xl max-w-2xl w-full text-gray-300 relative transform transition-all max-h-[90vh] overflow-y-auto"
                onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
            >
                <button 
                    onClick={onClose} 
                    className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors z-10"
                    aria-label="Close modal"
                >
                    <XIcon className="w-6 h-6" />
                </button>
                
                <h2 id="info-modal-title" className="text-2xl font-bold text-white mb-6 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-pink-500">{title}</h2>
                
                <div className="prose prose-invert prose-p:text-gray-300 prose-headings:text-gray-100 prose-strong:text-white prose-a:text-cyan-400 max-w-none">
                    {children}
                </div>
            </div>
        </div>
    );
};
