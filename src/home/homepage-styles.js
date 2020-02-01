import { Button } from 'reactstrap';
import BGImg from '../../public/deeplearning-background.jpg';
import BGCloud from '../../public/clouds.png';
import ProcessImg1 from '../../public/learningprocess.gif';
import ProcessImg2 from '../../public/dataprocessing.gif';
import ProcessImg3 from '../../public/networkprocessing.gif';
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
    transition: 300ms all ease;
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
    transition: 300ms all ease;
    transform: scale(0);
    opacity: 0;
`

const HomepageContainer = {
    main: {
        postion: 'relative',
        backgroundImage: `url(${BGImg})`,
        backgroundColor: '#0b1011',
        backgroundPosition: '50%',
        backgroundSize: 'cover',
        minHeight: '100vh',
        maxHeight: '999px',
        overflow: 'hidden',
        width: '100%',
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
    },
    canvasButtonBox: {
        position: 'relative',
        zIndex: 500,
        buttonFocus: 'none',
    },
    canvasButtons: {
        position: 'relative',
        margin: '0 5px',
        zIndex: 500,
    },
    canvasBox: {
        position: 'relative',
        zIndex: 500,
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
};


export { HomepageContainer, Clouds, StyledButtons, ImgProcessing1, ImgProcessing2, ImgProcessing3 };