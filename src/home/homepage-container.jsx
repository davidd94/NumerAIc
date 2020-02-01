import React, { useState, useRef } from 'react';
import HomepagePresent from './homepage-present.jsx';


const HomepageContainer = () => {
    const canvasRef = useRef();

    const [submitStatus, setSubmitStatus] = useState(false);
    
    const submitResults = () => {
        console.log('sending img data..');
        setSubmitStatus(true);
        setTimeout(() => {
            setSubmitStatus(false);
        }, 5000);
    };

    return (
        <>
            <HomepagePresent 
                    submitResults={submitResults}
                    submitStatus={submitStatus}
                    ref={canvasRef} />
        </>
    );
};


export default HomepageContainer