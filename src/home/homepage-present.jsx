import React, { forwardRef } from 'react';
import { Badge, InputGroup, InputGroupAddon, InputGroupText } from 'reactstrap';
import CanvasDraw from 'react-canvas-draw';
import { HomepageContainer, Clouds, StyledButtons,
         ImgProcessing1, ImgProcessing2, ImgProcessing3,
         ResponseIncorrectSpan, StyledInput,
         ImgTraining1, ImgTraining2,
} from './homepage-styles';


const HomepagePresent = forwardRef((props, ref) => {
    const canvasRef = ref.ref1;
    const dataResult = ref.ref2;

    return (
        <div style={HomepageContainer.main}>
            <div style={props.submitStatus || props.trainStatus ? {transform: 'scale(0)', opacity: '0'} : HomepageContainer.canvasContainer}>
                <div style={HomepageContainer.canvasOptionBox}>
                    <StyledButtons color="secondary" onClick={() => canvasRef.current.clear()}>Clear</StyledButtons>
                    <StyledButtons color="secondary" onClick={() => canvasRef.current.undo()}>Undo</StyledButtons>
                    <StyledButtons color={props.mode === "testing" ? "primary" : "danger"}
                                   onClick={props.mode === "testing" ? () => props.setCanvasMode('training') : () => props.setCanvasMode('testing')}
                                        >{props.mode} <Badge color="secondary" style={props.mode === 'training'? {marginLeft: '3px'} : {display: 'none'}}>{props.trainCt}</Badge>
                    </StyledButtons>
                </div>
                <CanvasDraw ref={canvasRef}
                            lazyRadius={0}
                            brushRadius={10}
                            brushColor={"#4efd54"}
                            gridColor={"#6df3ff"}
                            style={HomepageContainer.canvasBox} />
                <div style={HomepageContainer.canvasSubmitBox}>
                    <StyledButtons color={props.submitStatus ? "success" : "info"} onClick={() => props.submitResults()} style={{margin: '10px 10px'}} disabled={props.trainStatus || props.submitStatus}>{props.submitStatus ? 'Processing...' : 'Submit'}</StyledButtons>
                    <StyledButtons color={props.trainStatus ? "success" : "warning"} onClick={() => props.trainRequest()} style={props.mode === 'training' ? {margin: '10px 10px'} : {display: 'none'}} disabled={props.trainStatus || props.submitStatus}>{props.trainStatus ? 'Training...' : 'Train'}</StyledButtons>
                    <InputGroup style={props.mode === 'training' ? {width: '85px', margin: '0 10px'} : {display: 'none'}}>
                        <InputGroupAddon addonType="prepend">
                            <InputGroupText>#</InputGroupText>
                        </InputGroupAddon>
                        <StyledInput innerRef={dataResult} type="number" maxLength={1} style={{textAlign: 'center'}} onKeyUp={() => props.updateImgInput()} />
                    </InputGroup>
                </div>
            </div>
            <div style={props.submitStatus || props.trainStatus ? {...HomepageContainer.canvasImgContainer, transform: 'scale(1)', opacity: '0.75'} : HomepageContainer.canvasImgContainer}>
                <div style={HomepageContainer.canvasImg} />
            </div>
            <ImgProcessing1 style={props.submitStatus ? {transform: 'scale(1)', opacity: '0.85'} : {}}>
                <div style={HomepageContainer.processingImg1} />
            </ImgProcessing1>
            <ImgProcessing2 style={props.submitStatus ? {transform: 'scale(0.5)', opacity: '0.85'} : {}}>
                <div style={HomepageContainer.processingImg2} />
            </ImgProcessing2>
            <ImgProcessing3 style={props.submitStatus || props.trainStatus ? {transform: 'scale(1)', opacity: '0.9'} : {}}>
                <div style={HomepageContainer.processingImg3} />
            </ImgProcessing3>
            <div style={props.guess !== '' && !props.submitStatus ? {...HomepageContainer.responseBox, width: '400px', height: '200px', opacity: '1'} : HomepageContainer.responseBox}>
                <p style={HomepageContainer.responseText}>The Computer AI Guesses:</p>
                <p style={HomepageContainer.responseInnerText}>{props.guess}</p>
                <p style={HomepageContainer.responseIncorrectText}> Was I wrong ? <ResponseIncorrectSpan onClick={() => props.setCanvasMode('training')}>Re-train</ResponseIncorrectSpan> my brain !</p>
            </div>
            <ImgTraining1 style={props.trainStatus ? {transform: 'scale(0.9)', opacity: '0.9'} : {}}>
                <div style={HomepageContainer.trainingImg1} />
            </ImgTraining1>
            <ImgTraining2 style={props.trainStatus ? {transform: 'scale(1)', opacity: '0.9'} : {}}>
                <div style={HomepageContainer.trainingImg2} />
            </ImgTraining2>
            <Clouds />
        </div>
    );
});


export default HomepagePresent;