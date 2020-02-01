import React, { forwardRef } from 'react';
import { Button } from 'reactstrap';
import CanvasDraw from 'react-canvas-draw';
import { HomepageContainer, Clouds, StyledButtons, ImgProcessing1, ImgProcessing2, ImgProcessing3 } from './homepage-styles';


const HomepagePresent = forwardRef((props, ref) => {
    return (
        <div style={HomepageContainer.main}>
            <div style={HomepageContainer.canvasContainer}>
                <div style={HomepageContainer.canvasButtons}>
                    <StyledButtons color="secondary" onClick={() => ref.current.clear()}>Clear</StyledButtons>
                    <StyledButtons color="secondary" onClick={() => ref.current.undo()}>Undo</StyledButtons>
                </div>
                <CanvasDraw ref={ref}
                            lazyRadius={0}
                            brushRadius={10}
                            style={HomepageContainer.canvasBox} />
                <StyledButtons color={props.submitStatus ? "success" : "info"} onClick={() => props.submitResults()} style={{margin: '10px 0'}}>{props.submitStatus ? 'Processing...' : 'Submit'}</StyledButtons>
            </div>
            <ImgProcessing1 style={props.submitStatus ? {transform: 'scale(1)', opacity: '0.85'} : {}}>
                <div style={HomepageContainer.processingImg1} />
            </ImgProcessing1>
            <ImgProcessing2 style={props.submitStatus ? {transform: 'scale(0.5)', opacity: '0.85'} : {}}>
                <div style={HomepageContainer.processingImg2} />
            </ImgProcessing2>
            <ImgProcessing3 style={props.submitStatus ? {transform: 'scale(1)', opacity: '0.9'} : {}}>
                <div style={HomepageContainer.processingImg3} />
            </ImgProcessing3>
            <Clouds />
        </div>
    );
});


export default HomepagePresent;