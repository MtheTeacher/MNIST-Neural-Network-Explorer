
import React from 'react';

export const Header: React.FC = () => {
    return (
        <header className="w-full text-center">
            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-pink-500 to-orange-400">
                MNIST Neural Network Explorer
            </h1>
            <p className="mt-2 text-lg text-gray-300">
                Build, Train, and Test Your Own Digit Recognition Model
            </p>
        </header>
    );
};
