
import React from 'react';
import { XIcon } from '../constants';

interface AboutModalProps {
    onClose: () => void;
}

export const AboutModal: React.FC<AboutModalProps> = ({ onClose }) => {
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
            aria-labelledby="about-modal-title"
        >
            <div 
                className="bg-gray-900 border border-white/20 rounded-2xl p-8 shadow-2xl max-w-2xl w-full text-gray-300 relative transform transition-all"
                onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside
            >
                <button 
                    onClick={onClose} 
                    className="absolute top-4 right-4 p-2 text-gray-400 hover:text-white transition-colors"
                    aria-label="Close about modal"
                >
                    <XIcon className="w-6 h-6" />
                </button>
                
                <h2 id="about-modal-title" className="text-2xl font-bold text-white mb-6 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-pink-500">About This App</h2>
                
                <div className="space-y-4">
                    <p>
                        This app was developed by Morgan Andreasson for the purpose of teaching machine learning in a visual and engaging manner.
                    </p>
                    
                    <p>
                        The app was developed in Google's AI Studio with Gemini 2.5 Pro and with the aid of GPT-5-Codex for deep troubleshooting.
                    </p>
                </div>

                <div className="border-t border-white/20 pt-4 mt-6 text-sm text-gray-400">
                    <p>&copy; {new Date().getFullYear()} Morgan Andreasson</p>
                </div>
            </div>
        </div>
    );
};
