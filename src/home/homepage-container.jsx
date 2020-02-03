import React, { useState, useRef, useEffect } from 'react';
import HomepagePresent from './homepage-present.jsx';


const HomepageContainer = () => {
    const canvasRef = useRef();
    const dataResult = useRef();
    const refs = {
        ref1: canvasRef,
        ref2: dataResult
    };

    const [userID, setUserID] = useState('');
    const [mode, setMode] = useState('testing');
    const [submitStatus, setSubmitStatus] = useState(false);
    const [trainStatus, setTrainStatus] = useState(false);
    const [trainResult, setTrainResult] = useState('');
    const [guess, setGuess] = useState('');
    const [trainCt, setTrainCt] = useState(0);
    
    useEffect(() => {
        var randomUserID = Math.random().toString(36).substring(2, 16);
        setUserID(randomUserID);
    }, []);

    const submitResults = () => {
        var imgFile = canvasRef.current.canvasContainer.children[1].toDataURL();
        var url = (mode === 'training' ? '/saveusertraindata' : '/predict');
        var dataBody = (mode === 'training'? {'data': imgFile, 'dataResult': dataResult.current.value, 'userid': userID} : {'data': imgFile, 'userid': userID});
        
        if (trainResult === '' && mode === 'training' && (dataResult.current.value).length === 0) {
            alert("You have to enter your image's answer for me to learn from or how else would I know if it's right!");
        } else {
            setSubmitStatus(true);
            fetch(url, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dataBody)
            }).then(serverResponse => {
                serverResponse.json()
                .then((res) => {
                    setTimeout(() => {
                        setSubmitStatus(false);
                        if (res === 'User training data saved successfully') {
                            setTrainCt(trainCt + 1);
                            setTrainResult('');
                            dataResult.current.value = '';
                            canvasRef.current.clear();
                        } else if (Number.isInteger(res)) {
                            setGuess(res);
                            canvasRef.current.clear();
                        } else {
                            alert(res);
                        };
                    }, 5000);
                });
            });
        };
    };

    const trainRequest = () => {
        if (trainCt > 2) {
            setTrainStatus(true);
            var dataBody = {'userid': userID};

            fetch('/trainuserdata', {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dataBody)
            }).then(serverResponse => {
                serverResponse.json().then((res) => {
                    setTimeout(() => {
                        setTrainStatus(false);
                        setGuess('');
                        setMode('testing');
                        canvasRef.current.clear();
                        alert(res);
                    }, 8000);
                });
            });
        } else {
            alert('I cannot learn without at least 3 new data samples! Please draw and submit some samples first.')
        };
    };

    const setCanvasMode = (mode) => {
        setMode(mode);
        setGuess('');
    };
    
    const updateImgInput = () => {
        if ((dataResult.current.value).toString().length <= 1) {
            setTrainResult(dataResult.current.value);
        } else {
            var sliced = (dataResult.current.value).toString().slice(1, 2);
            dataResult.current.value = sliced;
            setTrainResult(sliced);
        };
    };

    return (
        <>
            <HomepagePresent 
                    submitResults={submitResults}
                    submitStatus={submitStatus}
                    trainRequest={trainRequest}
                    trainStatus={trainStatus}
                    updateImgInput={updateImgInput}
                    setCanvasMode={setCanvasMode}
                    mode={mode}
                    guess={guess}
                    trainCt={trainCt}
                    ref={refs} />
        </>
    );
};


export default HomepageContainer