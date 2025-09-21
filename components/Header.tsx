
import React from 'react';
import { InfoIcon } from '../constants';

interface HeaderProps {
    onShowAbout: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onShowAbout }) => {
    return (
        <header className="w-full text-center relative">
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-pink-500 to-orange-400">
                MNIST Neural Network Explorer
            </h1>
            <p className="mt-2 text-lg text-gray-300">
                Build, Train, and Test Your Own Digit Recognition Model
            </p>
            <button 
                onClick={onShowAbout} 
                className="absolute top-0 right-0 p-2 text-gray-400 hover:text-cyan-300 transition-colors"
                aria-label="About this application"
            >
                <InfoIcon className="w-6 h-6" />
            </button>
        </header>
    );
};
