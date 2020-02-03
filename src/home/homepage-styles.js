import { Button, Input } from 'reactstrap';
import BGImg from '../../public/deeplearning-background3.gif';
import BGCloud from '../../public/clouds.png';
import ProcessImg1 from '../../public/learningprocess.gif';
import ProcessImg2 from '../../public/dataprocessing.gif';
import ProcessImg3 from '../../public/networkprocessing.gif';
import CanvasImg from '../../public/brainprocessing.gif';
import TrainImg1 from '../../public/cnnscan3.gif';
import TrainImg2 from '../../public/cnnrotation.gif';
import styled, { keyframes } from 'styled-components';


const cloudKeyframes = keyframes`
    0% { transform: translateZ(0); }
    100% { transform: translate3d(-50%,0,0); }
`;

const imgKeyframes1 = keyframes`
    0% { right: 400px; top: -100px; }
    20% { right: 350px; top: -120px; }
    40% { right: 300px; top: -140px; }
    60% { right: 350px; top: -140px; }
    80% { right: 375px; top: -120px; }
    100% { right: 400px; top: -100px; }
`

const imgKeyframes2 = keyframes`
    0% { right: 500px; bottom: 0px; }
    20% { right: 490px; bottom: 5px; }
    40% { right: 495px; bottom: 15px; }
    60% { right: 505px; bottom: 10px; }
    80% { right: 503px; bottom: 5px; }
    100% { right: 500px; bottom: 0px; }
`

const imgKeyframes3 = keyframes`
    0% { right: 600px; bottom: 0px; }
    10% { right: 595px; bottom: 8px; }
    20% { right: 590px; bottom: 5px; }
    30% { right: 600px; bottom: 15px; }
    40% { right: 595px; bottom: 15px; }
    50% { right: 590px; bottom: 5px; }
    60% { right: 605px; bottom: 10px; }
    70% { right: 603px; bottom: 5px; }
    80% { right: 595px; bottom: 0px; }
    90% { right: 595px; bottom: 2px; }
    100% { right: 600px; bottom: 0px; }
`

const imgKeyframes4 = keyframes`
    0% { right: 250px; top: 100px; }
    10% { right: 225px; top: 125px; }
    20% { right: 200px; top: 150px; }
    30% { right: 225px; top: 125px; }
    40% { right: 300px; top: 125px; }
    50% { right: 375px; top: 100px; }
    60% { right: 300px; top: 15px; }
    70% { right: 275px; top: 150px; }
    80% { right: 290px; top: 125px; }
    90% { right: 270px; top: 110px; }
    100% { right: 250px; top: 100px; }
`

const Clouds = styled.div`
    position: absolute;
    background-image: url(${BGCloud});
    z-index: 2;
    bottom: 0;
    left: 0;
    width: 250.625em;
    height: 100%;
    animation: ${cloudKeyframes};
    animation-duration: 80s;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
`

const StyledButtons = styled(Button)`
    position: relative;
    font-family: 'Orbitron, Arial, san-serif';
    text-transform: uppercase;
    margin: 0 5px;
    z-index: 500;
    :focus {
        outline: none !important;
        box-shadow: none;
    };
    :active {
        outline: none !important;
        box-shadow: none;
    };
`

const StyledInput = styled(Input)`
    ::-webkit-inner-spin-button{
        -webkit-appearance: none; 
        margin: 0; 
    };
    ::-webkit-outer-spin-button{
        -webkit-appearance: none; 
        margin: 0; 
    };
`

const ImgProcessing1 = styled.div`
    animation: ${imgKeyframes1};
    animation-duration: 8s;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    position: absolute;
    z-index: 500;
    top: -100px;
    right: 400px;
    transition: 300ms all ease;
    transform: scale(0);
    opacity: 0;
`

const ImgProcessing2 = styled.div`
    animation: ${imgKeyframes2};
    animation-duration: 8s;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    position: absolute;
    bottom: 0;
    right: 500px;
    z-index: 500;
    border-radius: 50%;
    width: 500px;
    height: 500px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: black;
    transition: 900ms all ease;
    transform: scale(0);
    opacity: 0;
`

const ImgProcessing3 = styled.div`
    position: absolute;
    bottom: 50px;
    right: 100px;
    z-index: 500;
    border-radius: 50%;
    width: 400px;
    height: 400px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: black;
    transition: 1500ms all ease;
    transform: scale(0);
    opacity: 0;
`

const ImgTraining1 = styled.div`
    animation: ${imgKeyframes3};
    animation-duration: 8s;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    position: absolute;
    bottom: 0px;
    right: 600px;
    z-index: 500;
    border-radius: 50%;
    width: 350px;
    height: 350px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: black;
    transition: 1500ms all ease;
    transform: scale(0);
    opacity: 0;
`

const ImgTraining2 = styled.div`
    animation: ${imgKeyframes4};
    animation-duration: 8s;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    position: absolute;
    top: 100px;
    right: 250px;
    z-index: 500;
    border-radius: 50%;
    width: 400px;
    height: 400px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: black;
    transition: 1500ms all ease;
    transform: scale(0);
    opacity: 0;
`

const ResponseIncorrectSpan = styled.span`
    cursor: pointer;
    color: #007bff;
    :hover {
        text-decoration: underline;
        color: #004c9e;
    };
`

const HomepageContainer = {
    main: {
        position: 'relative',
        backgroundImage: `url(${BGImg})`,
        backgroundColor: '#0b1011',
        backgroundRepeat: 'no-repeat',
        backgroundSize: 'cover',
        backgroundPosition: 'right',
        minHeight: '100vh',
        maxHeight: '999px',
        overflow: 'hidden',
        width: '100vw',
        zIndex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    canvasContainer: {
        position: 'absolute',
        top: '200px',
        left: '100px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        transition: '800ms all ease',
        transform: 'scale(1)',
        opacity: '1',
        zIndex: '500',
    },
    canvasOptionBox: {
        pposition: 'relative',
        margin: '0 5px',
        zIndex: 500,
    },
    canvasSubmitBox: {
        position: 'relative',
        margin: '0 20px',
        zIndex: 500,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    canvasBox: {
        position: 'relative',
        zIndex: 500,
        backgroundColor: 'rgb(40, 1, 55)',
    },
    canvasImgContainer: {
        position: 'absolute',
        top: '200px',
        left: '100px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: '500ms all ease',
        transform: 'scale(0)',
        opacity: '0',
        zIndex: '600',
    },
    canvasImg: {
        position: 'relative',
        transform: 'scale(1)',
        width: '450px',
        height: '450px',
        backgroundImage: `url(${CanvasImg})`,
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        borderRadius: '10px',
    },
    processingImg1: {
        position: 'relative',
        backgroundImage: `url(${ProcessImg1})`,
        width: '500px',
        height: '500px',
        borderRadius: '50%',
        transform: 'scale(0.50)',
    },
    processingImg2: {
        position: 'absolute',
        backgroundImage: `url(${ProcessImg2})`,
        width: '400px',
        height: '400px',
        bottom: '-25px',
        backgroundRepeat: 'no-repeat',
        backgroundSize: '100%',
    },
    processingImg3: {
        position: 'absolute',
        backgroundImage: `url(${ProcessImg3})`,
        width: '400px',
        height: '400px',
        bottom: '25px',
        backgroundRepeat: 'no-repeat',
        backgroundSize: '100%',
        backgroundPosition: 'center',
    },
    trainingImg1: {
        position: 'absolute',
        backgroundImage: `url(${TrainImg1})`,
        width: '300px',
        height: '300px',
        bottom: '25px',
        backgroundRepeat: 'no-repeat',
        backgroundSize: '100%',
        backgroundPosition: 'center',
        borderRadius: '15px',
    },
    trainingImg2: {
        position: 'absolute',
        backgroundImage: `url(${TrainImg2})`,
        width: '350px',
        height: '350px',
        bottom: '25px',
        backgroundRepeat: 'no-repeat',
        backgroundSize: '100%',
        backgroundPosition: 'center',
    },
    responseBox: {
        position: 'absolute',
        top: '50%',
        left: '60%',
        width: '1px',
        height: '50px',
        borderRadius: '5px',
        backgroundColor: 'black',
        transform: 'translate(-50%, -50%)',
        opacity: '0',
        transition: '3000ms all ease',
        zIndex: '350',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 30px',
        overflow: 'hidden',
    },
    responseText: {
        color: 'white',
        fontSize: '18px',
        textAlign: 'center',
        fontFamily: 'Orbitron, Arial, san-serif',
        margin: '0 5px',
        width: '100%',
    },
    responseInnerText: {
        color: '#39ff14',
        textAlign: 'center',
        fontSize: '28px',
        fontFamily: 'Orbitron, Arial, san-serif',
        margin: '0 5px',
    },
    responseIncorrectText: {
        position: 'fixed',
        bottom: '10px',
        color: 'rgb(238, 77, 220)',
        fontSize: '14px',
        textAlign: 'center',
        fontFamily: 'Orbitron, Arial, san-serif',
    },
};


export { HomepageContainer, Clouds, StyledButtons,
        ImgProcessing1, ImgProcessing2, ImgProcessing3,
        ResponseIncorrectSpan, StyledInput,
        ImgTraining1, ImgTraining2 };