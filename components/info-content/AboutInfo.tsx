
import React from 'react';

export const AboutInfo: React.FC = () => {
    return (
        <>
            <p>
                This app was developed by Morgan Andreasson for the purpose of teaching machine learning in a visual and engaging manner.
            </p>
            
            <p>
                The app was developed in Google's AI Studio with Gemini 2.5 Pro and with the aid of GPT-5-Codex for deep troubleshooting.
            </p>

            <div className="border-t border-white/20 pt-4 mt-6 text-sm text-gray-400">
                <p>&copy; {new Date().getFullYear()} Morgan Andreasson</p>
            </div>
        </>
    );
};
