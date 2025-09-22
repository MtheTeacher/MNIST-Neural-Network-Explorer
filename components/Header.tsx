
import React from 'react';
import { InfoIcon } from '../constants';

type Page = 'mnist' | 'wavescape';

interface HeaderProps {
    onShowInfo: (topic: string) => void;
    page: Page;
    setPage: (page: Page) => void;
}

export const Header: React.FC<HeaderProps> = ({ onShowInfo, page, setPage }) => {
    const mainTitle = page === 'mnist' ? 'MNIST Neural Network Explorer' : 'Gradient Descent Explorer';
    const subTitle = page === 'mnist' ? 'Build, Train, and Test Your Own Digit Recognition Model' : 'Visualize Optimization in a 2D Landscape';

    return (
        <header className="w-full text-center relative">
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-pink-500 to-orange-400">
                {mainTitle}
            </h1>
            <p className="mt-2 text-lg text-gray-300">
                {subTitle}
            </p>
             <nav className="flex justify-center gap-2 sm:gap-4 mt-6">
                <button
                    onClick={() => setPage('mnist')}
                    className={`px-4 py-2 text-sm sm:text-base rounded-full font-semibold transition-all duration-300 transform hover:scale-105 ${
                        page === 'mnist'
                        ? 'bg-cyan-500 text-white shadow-lg'
                        : 'bg-white/10 text-gray-300 hover:bg-white/20'
                    }`}
                    aria-current={page === 'mnist'}
                >
                    MNIST Explorer
                </button>
                <button
                    onClick={() => setPage('wavescape')}
                    className={`px-4 py-2 text-sm sm:text-base rounded-full font-semibold transition-all duration-300 transform hover:scale-105 ${
                        page === 'wavescape'
                        ? 'bg-cyan-500 text-white shadow-lg'
                        : 'bg-white/10 text-gray-300 hover:bg-white/20'
                    }`}
                     aria-current={page === 'wavescape'}
                >
                    Gradient Descent
                </button>
            </nav>
            <button 
                onClick={() => onShowInfo('about')} 
                className="absolute top-0 right-0 p-2 text-gray-400 hover:text-cyan-300 transition-colors"
                aria-label="About this application"
            >
                <InfoIcon className="w-6 h-6" />
            </button>
        </header>
    );
};
