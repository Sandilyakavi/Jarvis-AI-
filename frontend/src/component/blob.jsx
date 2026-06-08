import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const noiseFunctions = `
  vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
  vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
  vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
  float snoise(vec3 v){
    const vec2 C=vec2(1.0/6.0,1.0/3.0);
    const vec4 D=vec4(0.0,0.5,1.0,2.0);
    vec3 i=floor(v+dot(v,C.yyy));
    vec3 x0=v-i+dot(i,C.xxx);
    vec3 g=step(x0.yzx,x0.xyz);
    vec3 l=1.0-g;
    vec3 i1=min(g.xyz,l.zxy);
    vec3 i2=max(g.xyz,l.zxy);
    vec3 x1=x0-i1+C.xxx;
    vec3 x2=x0-i2+C.yyy;
    vec3 x3=x0-D.yyy;
    i=mod289(i);
    vec4 p=permute(permute(permute(
      i.z+vec4(0.0,i1.z,i2.z,1.0))
      +i.y+vec4(0.0,i1.y,i2.y,1.0))
      +i.x+vec4(0.0,i1.x,i2.x,1.0));
    float n_=0.142857142857;
    vec3 ns=n_*D.wyz-D.xzx;
    vec4 j=p-49.0*floor(p*ns.z*ns.z);
    vec4 x_=floor(j*ns.z);
    vec4 y_=floor(j-7.0*x_);
    vec4 x=x_*ns.x+ns.yyyy;
    vec4 y=y_*ns.x+ns.yyyy;
    vec4 h=1.0-abs(x)-abs(y);
    vec4 b0=vec4(x.xy,y.xy);
    vec4 b1=vec4(x.zw,y.zw);
    vec4 s0=floor(b0)*2.0+1.0;
    vec4 s1=floor(b1)*2.0+1.0;
    vec4 sh=-step(h,vec4(0.0));
    vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;
    vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
    vec3 p0=vec3(a0.xy,h.x);
    vec3 p1=vec3(a0.zw,h.y);
    vec3 p2=vec3(a1.xy,h.z);
    vec3 p3=vec3(a1.zw,h.w);
    vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
    p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
    vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);
    m=m*m;
    return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
  }
  float fbm(vec3 p){
    float total=0.0,amplitude=0.5,frequency=1.0;
    for(int i=0;i<3;i++){
      total+=snoise(p*frequency)*amplitude;
      amplitude*=0.5;frequency*=2.0;
    }
    return total;
  }
`;

export default function PlasmaSphere({ blobTheme = 'amber', blobSize = 1.0, blobPosition = { x: 0, y: -0.75 }, setBlobPosition, isDraggingMode = false }) {
    const mountRef = useRef(null);
    const audioRef = useRef({ analyser: null, dataArray: null, ctx: null, source: null });
    const scaleRef = useRef(1.0);
    const smoothScaleRef = useRef(1.0);
    const [listening, setListening] = useState(false);
    const [permissionDenied, setPermissionDenied] = useState(false);
    const [finalTranscript, setFinalTranscript] = useState('');
    const [interimTranscript, setInterimTranscript] = useState('');
    const [jarvisResponse, setJarvisResponse] = useState('');
    const [isThinking, setIsThinking] = useState(false);
    const [wakeListening, setWakeListening] = useState(false);
    const [wakeDetected, setWakeDetected] = useState(false);
    const recognitionRef = useRef(null);
    const wakeRef = useRef(null);
    const listeningRef = useRef(false);
    const lastProcessedTranscriptRef = useRef(''); // Prevent duplicate processing

    // The GROQ API key is now securely handled by the backend server.

    // Derive the display transcript — interim overrides final while speaking
    const transcript = interimTranscript || finalTranscript;

    const stateRef = useRef({ blobSize, blobTheme, isDraggingMode, blobPosition, listening });

    useEffect(() => {
        stateRef.current = { blobSize, blobTheme, isDraggingMode, blobPosition, listening };
        listeningRef.current = listening;
    }, [blobSize, blobTheme, isDraggingMode, blobPosition, listening]);

    useEffect(() => {
        return () => {
            recognitionRef.current?.stop();
            wakeRef.current?.stop();
        };
    }, []);

    useEffect(() => {
        if (!mountRef.current) return;

        let renderer, animFrameId;

        function init() {
            const W = mountRef.current.clientWidth;
            const H = mountRef.current.clientHeight;

            const scene = new THREE.Scene();

            const camera = new THREE.PerspectiveCamera(75, W / H, 0.1, 100);
            camera.position.z = 2.4;

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(W, H);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 0.9;
            mountRef.current.innerHTML = ''; // Clear any existing duplicate canvases from HMR/StrictMode
            mountRef.current.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.enablePan = false;
            controls.minDistance = 1.5;
            controls.maxDistance = 20;

            const mainGroup = new THREE.Group();
            mainGroup.position.x = stateRef.current.blobPosition.x;
            mainGroup.position.y = stateRef.current.blobPosition.y;
            scene.add(mainGroup);

            const hitPlane = new THREE.Mesh(
                new THREE.PlaneGeometry(100, 100),
                new THREE.MeshBasicMaterial({ visible: false })
            );
            scene.add(hitPlane);

            // Shell
            const shellGeo = new THREE.SphereGeometry(1.0, 64, 64);
            const shellVS = `
        varying vec3 vNormal,vViewPosition;
        void main(){
          vNormal=normalize(normalMatrix*normal);
          vec4 mvPosition=modelViewMatrix*vec4(position,1.0);
          vViewPosition=-mvPosition.xyz;
          gl_Position=projectionMatrix*mvPosition;
        }`;
            const shellFS = `
        varying vec3 vNormal,vViewPosition;
        uniform vec3 uColor;uniform float uOpacity;
        void main(){
          float fresnel=pow(1.0-dot(normalize(vNormal),normalize(vViewPosition)),2.5);
          gl_FragColor=vec4(uColor,fresnel*uOpacity);
        }`;

            const shellBackMat = new THREE.ShaderMaterial({
                vertexShader: shellVS, fragmentShader: shellFS,
                uniforms: { uColor: { value: new THREE.Color(0x001622) }, uOpacity: { value: 0.2 } },
                transparent: true, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false
            });
            const shellFrontMat = new THREE.ShaderMaterial({
                vertexShader: shellVS, fragmentShader: shellFS,
                uniforms: { uColor: { value: new THREE.Color(0xff6600) }, uOpacity: { value: 0.2 } },
                transparent: true, blending: THREE.AdditiveBlending, side: THREE.FrontSide, depthWrite: false
            });
            mainGroup.add(new THREE.Mesh(shellGeo, shellBackMat));
            mainGroup.add(new THREE.Mesh(shellGeo, shellFrontMat));

            // Plasma
            const plasmaGeo = new THREE.SphereGeometry(0.998, 128, 128);
            const plasmaMat = new THREE.ShaderMaterial({
                uniforms: {
                    uTime: { value: 0 },
                    uScale: { value: 0.2 },
                    uBrightness: { value: 1.1 },
                    uThreshold: { value: 0.09 },
                    uColorDeep: { value: new THREE.Color(0x001b2a) },
                    uColorMid: { value: new THREE.Color(0xd45500) },
                    uColorBright: { value: new THREE.Color(0xffaa00) },
                    uMicEnergy: { value: 0.0 }
                },
                vertexShader: `
          uniform float uMicEnergy;
          varying vec3 vPosition,vNormal,vViewPosition;
          ${noiseFunctions}
          void main(){
            vPosition=position;
            vNormal=normalize(normalMatrix*normal);
            vec3 displaced=position;
            float disp=snoise(position*3.0+uMicEnergy*0.5)*uMicEnergy*0.18;
            displaced+=normal*disp;
            vec4 mvPosition=modelViewMatrix*vec4(displaced,1.0);
            vViewPosition=-mvPosition.xyz;
            gl_Position=projectionMatrix*mvPosition;
          }`,
                fragmentShader: `
          uniform float uTime,uScale,uBrightness,uThreshold,uMicEnergy;
          uniform vec3 uColorDeep,uColorMid,uColorBright;
          varying vec3 vPosition,vNormal,vViewPosition;
          ${noiseFunctions}
          void main(){
            vec3 p=vPosition*uScale;
            vec3 q=vec3(
              fbm(p+vec3(0.0,uTime*0.05,0.0)),
              fbm(p+vec3(5.2,1.3,2.8)+uTime*0.05),
              fbm(p+vec3(2.2,8.4,0.5)-uTime*0.02)
            );
            float density=fbm(p+2.0*q);
            float t=(density+0.4)*0.8;
            float extraBright=1.0+uMicEnergy*1.5;
            float alpha=smoothstep(uThreshold,0.7,t);
            vec3 cWhite=vec3(1.0,1.0,1.0);
            vec3 color=mix(uColorDeep,uColorMid,smoothstep(uThreshold,0.5,t));
            color=mix(color,uColorBright,smoothstep(0.5,0.8,t));
            color=mix(color,cWhite,smoothstep(0.8,1.0,t));
            float facing=dot(normalize(vNormal),normalize(vViewPosition));
            float depthFactor=(facing+1.0)*0.5;
            float finalAlpha=alpha*(0.02+0.98*depthFactor);
            gl_FragColor=vec4(color*uBrightness*extraBright,finalAlpha);
          }`,
                transparent: true, blending: THREE.AdditiveBlending,
                side: THREE.DoubleSide, depthWrite: false
            });
            const plasmaMesh = new THREE.Mesh(plasmaGeo, plasmaMat);
            mainGroup.add(plasmaMesh);

            // Particles
            const pCount = 600;
            const pPos = new Float32Array(pCount * 3);
            const pSizes = new Float32Array(pCount);
            for (let i = 0; i < pCount; i++) {
                const r = 0.95 * Math.cbrt(Math.random());
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                pPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
                pPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
                pPos[i * 3 + 2] = r * Math.cos(phi);
                pSizes[i] = Math.random();
            }
            const pGeo = new THREE.BufferGeometry();
            pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
            pGeo.setAttribute("aSize", new THREE.BufferAttribute(pSizes, 1));
            const pMat = new THREE.ShaderMaterial({
                uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color(0xffdd66) } },
                vertexShader: `
          uniform float uTime;attribute float aSize;varying float vAlpha;
          void main(){
            vec3 pos=position;
            pos.y+=sin(uTime*0.2+pos.x)*0.02;
            pos.x+=cos(uTime*0.15+pos.z)*0.02;
            vec4 mvPosition=modelViewMatrix*vec4(pos,1.0);
            gl_Position=projectionMatrix*mvPosition;
            gl_PointSize=(8.0*aSize+4.0)*(1.0/-mvPosition.z);
            vAlpha=0.8+0.2*sin(uTime+aSize*10.0);
          }`,
                fragmentShader: `
          uniform vec3 uColor;varying float vAlpha;
          void main(){
            vec2 uv=gl_PointCoord-vec2(0.5);
            float dist=length(uv);
            if(dist>0.5)discard;
            float glow=pow(1.0-dist*2.0,1.8);
            gl_FragColor=vec4(uColor,glow*vAlpha);
          }`,
                transparent: true, blending: THREE.AdditiveBlending, depthWrite: false
            });
            mainGroup.add(new THREE.Points(pGeo, pMat));

            const themes = {
                amber: { deep: 0x001b2a, mid: 0xd45500, bright: 0xffaa00, front: 0xff6600, back: 0x001622, particles: 0xffdd66 },
                cyan: { deep: 0x000b14, mid: 0x006688, bright: 0x00ffff, front: 0x00aaff, back: 0x00111a, particles: 0x88ffff },
                emerald: { deep: 0x001a11, mid: 0x008844, bright: 0x00ff88, front: 0x00ff66, back: 0x00110a, particles: 0x88ffaa },
                crimson: { deep: 0x1a0000, mid: 0x880022, bright: 0xff0044, front: 0xff0033, back: 0x110000, particles: 0xff88aa }
            };

            const clock = new THREE.Clock();

            // Dragging Logic
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            let isDragging = false;
            let dragOffset = new THREE.Vector3();

            const onPointerDown = (event) => {
                if (!stateRef.current.isDraggingMode) return;
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObject(hitPlane);
                if (intersects.length > 0) {
                    isDragging = true;
                    dragOffset.copy(intersects[0].point).sub(mainGroup.position);
                    controls.enabled = false;
                }
            };

            const onPointerMove = (event) => {
                if (!isDragging || !stateRef.current.isDraggingMode) return;
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObject(hitPlane);
                if (intersects.length > 0) {
                    mainGroup.position.copy(intersects[0].point).sub(dragOffset);
                }
            };

            const onPointerUp = () => {
                if (isDragging) {
                    isDragging = false;
                    controls.enabled = true;
                    if (setBlobPosition) {
                        setBlobPosition({ x: mainGroup.position.x, y: mainGroup.position.y });
                    }
                }
            };

            renderer.domElement.addEventListener('pointerdown', onPointerDown);
            renderer.domElement.addEventListener('pointermove', onPointerMove);
            renderer.domElement.addEventListener('pointerup', onPointerUp);
            // also listen on window so if pointer leaves canvas it stops dragging
            window.addEventListener('pointerup', onPointerUp);

            function getAudioEnergy() {
                const { analyser, dataArray } = audioRef.current;
                if (!analyser || !dataArray) return 0;
                analyser.getByteFrequencyData(dataArray);
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
                return sum / (dataArray.length * 255);
            }

            function animate() {
                animFrameId = requestAnimationFrame(animate);
                const t = clock.getElapsedTime();

                const state = stateRef.current;

                // Sync Theme
                const theme = themes[state.blobTheme] || themes.amber;
                plasmaMat.uniforms.uColorDeep.value.setHex(theme.deep);
                plasmaMat.uniforms.uColorMid.value.setHex(theme.mid);
                plasmaMat.uniforms.uColorBright.value.setHex(theme.bright);
                shellFrontMat.uniforms.uColor.value.setHex(theme.front);
                shellBackMat.uniforms.uColor.value.setHex(theme.back);
                pMat.uniforms.uColor.value.setHex(theme.particles);

                controls.enabled = !state.isDraggingMode && !isDragging;

                if (!isDragging && !state.isDraggingMode) {
                    mainGroup.position.x += (state.blobPosition.x - mainGroup.position.x) * 0.1;
                    mainGroup.position.y += (state.blobPosition.y - mainGroup.position.y) * 0.1;
                }

                // Audio reactive scale
                const rawEnergy = getAudioEnergy();
                scaleRef.current = (0.45 * state.blobSize) + (rawEnergy * 2.8 * state.blobSize); // Grew sensitivity here (2.8 instead of 0.8)
                smoothScaleRef.current += (scaleRef.current - smoothScaleRef.current) * 0.12;
                mainGroup.scale.setScalar(smoothScaleRef.current);

                plasmaMat.uniforms.uTime.value = t * 1.2;
                plasmaMat.uniforms.uMicEnergy.value = rawEnergy;
                pMat.uniforms.uTime.value = t;

                plasmaMesh.rotation.y = t * 0.08;
                mainGroup.rotation.x += 0.002;
                mainGroup.rotation.y += 0.005;

                controls.update();
                renderer.render(scene, camera);
            }
            animate();

            const handleResize = () => {
                if (!mountRef.current) return;
                const w = mountRef.current.clientWidth;
                const h = mountRef.current.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            };
            window.addEventListener("resize", handleResize);

            return () => {
                window.removeEventListener("resize", handleResize);
                renderer.domElement.removeEventListener('pointerdown', onPointerDown);
                renderer.domElement.removeEventListener('pointermove', onPointerMove);
                renderer.domElement.removeEventListener('pointerup', onPointerUp);
                window.removeEventListener('pointerup', onPointerUp);
                cancelAnimationFrame(animFrameId);
                renderer.dispose();
                if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
                    mountRef.current.removeChild(renderer.domElement);
                }
            };
        }

        const cleanup = init();
        return () => { cleanup && cleanup(); };
    }, []);

    // ── Wake-word helpers ──────────────────────────────────────────
    const wakeWasActiveRef = useRef(false); // remember if wake was on before main listening

    function stopWakeListener({ silent = false } = {}) {
        if (wakeRef.current) {
            wakeRef.current.onend = null; // prevent auto-restart loop
            wakeRef.current.onresult = null;
            try { wakeRef.current.stop(); } catch (e) {}
            wakeRef.current = null;
        }
        if (!silent) setWakeListening(false);
    }

    function startWakeListener() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return;
        // Don't start if main listening is already active
        if (listeningRef.current) {
            wakeWasActiveRef.current = true; // remember to resume later
            setWakeListening(true);           // show the badge
            return;
        }

        const wake = new SpeechRecognition();
        wake.continuous = true;
        wake.interimResults = true;
        wake.lang = 'en-US';
        wake.maxAlternatives = 3;

        wake.onresult = (event) => {
            if (listeningRef.current) return;
            for (let i = 0; i < event.results.length; i++) {
                for (let j = 0; j < event.results[i].length; j++) {
                    const word = event.results[i][j].transcript.trim().toLowerCase();
                    if (word.includes('jarvis')) {
                        setWakeDetected(true);
                        setTimeout(() => setWakeDetected(false), 2500);
                        // Stop wake FIRST, then start main (browser only allows one at a time)
                        stopWakeListener({ silent: true });
                        wakeWasActiveRef.current = true; // resume wake when done
                        setTimeout(() => startListening(), 150); // tiny delay for browser
                        return;
                    }
                }
            }
        };

        wake.onend = () => {
            // Auto-restart only if still supposed to be running
            if (wakeRef.current === wake) {
                try { wake.start(); } catch (e) {}
            }
        };

        try {
            wake.start();
            wakeRef.current = wake;
            setWakeListening(true);
        } catch (e) {
            console.warn('Wake listener failed to start:', e);
        }
    }

    function toggleWake() {
        if (wakeListening) {
            wakeWasActiveRef.current = false;
            stopWakeListener();
        } else {
            startWakeListener();
        }
    }

    // ── Full-listening helpers ─────────────────────────────────────
    // Extracted so it can be called from both startListening and the language-change effect
    function startRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition || !listeningRef.current) return;

        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            let interim = '';
            let newFinal = '';
            // KEY FIX: start from event.resultIndex, not 0
            // This prevents re-reading results from a previous session/language
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const text = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    newFinal += text + ' ';
                } else {
                    interim += text;
                }
            }
            // Accumulate final words correctly
            if (newFinal.trim()) {
                const fullFinal = (finalTranscript ? finalTranscript + ' ' + newFinal.trim() : newFinal.trim());
                setFinalTranscript(fullFinal);
                
                // TRIGGER AI: If we have a substantial sentence, send it to Groq
                // We use a small timeout to see if more text is coming
                if (fullFinal.length > 3) {
                    clearTimeout(window.aiTriggerTimeout);
                    window.aiTriggerTimeout = setTimeout(() => {
                        handleConversation(fullFinal);
                    }, 1200); 
                }
            }
            setInterimTranscript(interim);
        };

        recognition.onerror = (e) => {
            // 'no-speech' is benign, skip logging
            if (e.error !== 'aborted' && e.error !== 'no-speech') {
                console.warn('Recognition error:', e.error);
            }
        };

        recognition.onend = () => {
            // Only auto-restart if this is still the active recognition instance
            if (listeningRef.current && recognitionRef.current === recognition) {
                try { recognition.start(); } catch (e) {}
            }
        };

        try { recognition.start(); } catch (e) {}
        recognitionRef.current = recognition;
    }

    async function startListening() {
        if (listeningRef.current) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const ctx = new AudioContext();
            const source = ctx.createMediaStreamSource(stream);
            const analyser = ctx.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            source.connect(analyser);
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            audioRef.current = { analyser, dataArray, ctx, source };
            setListening(true);
            listeningRef.current = true;
            setPermissionDenied(false);
            startRecognition();
        } catch {
            setPermissionDenied(true);
        }
    }

    // ── AI & TTS Logic ──────────────────────────────────────────
    async function handleConversation(userText) {
        if (!userText || userText === lastProcessedTranscriptRef.current) return;
        lastProcessedTranscriptRef.current = userText;
        
        setIsThinking(true);
        setJarvisResponse("Processing protocol...");

        try {
            const response = await fetch("https://jarvis-ai-ug9h.onrender.com/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: userText
                })
            });

            const data = await response.json();
            const reply = data.reply || "System error in response module.";
            
            setJarvisResponse(reply);
            setIsThinking(false);
            speak(reply);
        } catch (error) {
            console.error("Backend Error:", error);
            setJarvisResponse("Connection to neural core failed, Sir.");
            setIsThinking(false);
        }
    }

    function speak(text) {
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel(); // Stop current speech

        const utterance = new SpeechSynthesisUtterance(text);
        
        // Find the most humanoid voice
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => 
            v.name.includes("Neural") || 
            v.name.includes("Google US English") || 
            v.name.includes("Samantha") ||
            v.name.includes("Daniel")
        ) || voices[0];

        utterance.voice = preferredVoice;
        utterance.rate = 1.05; // Slightly faster for efficiency
        utterance.pitch = 0.95; // Slightly lower for a more mature sci-fi tone
        
        window.speechSynthesis.speak(utterance);
    }

    async function toggleMic() {
        if (listening) {
            // Stop main listening
            listeningRef.current = false;
            if (recognitionRef.current) {
                recognitionRef.current.onend = null;
                recognitionRef.current.onerror = null;
                try { recognitionRef.current.stop(); } catch (e) {}
                recognitionRef.current = null;
            }
            audioRef.current.ctx?.close();
            audioRef.current = { analyser: null, dataArray: null, ctx: null, source: null };
            setListening(false);
            setFinalTranscript('');
            setInterimTranscript('');
            setJarvisResponse('');
            lastProcessedTranscriptRef.current = '';
            // Resume wake listener if it was active before
            if (wakeWasActiveRef.current) {
                setTimeout(() => startWakeListener(), 200);
            }
            return;
        }
        // If wake was on, stop it first to free the audio engine
        if (wakeRef.current) stopWakeListener({ silent: true });
        await startListening();
    }

    return (
        <div style={{ width: "100%", height: "100vh", background: "transparent", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", fontFamily: "system-ui, sans-serif" }}>
            <div ref={mountRef} style={{ width: "100%", flex: 1, position: "relative", cursor: isDraggingMode ? 'move' : 'default' }} />

            {/* Wake-word indicator — top left */}
            <div style={{ position: "absolute", top: 80, left: 30, display: "flex", alignItems: "center", gap: 10, zIndex: 200 }}>
                <button
                    onClick={toggleWake}
                    title={wakeListening ? 'Wake word active — say JARVIS to start' : 'Click to enable wake word'}
                    style={{
                        display: "flex", alignItems: "center", gap: 8,
                        padding: "7px 16px",
                        background: wakeListening ? "rgba(0,255,225,0.08)" : "rgba(255,255,255,0.04)",
                        border: wakeListening ? "1px solid rgba(0,255,225,0.45)" : "1px solid rgba(255,255,255,0.12)",
                        clipPath: "polygon(10px 0, 100% 0, calc(100% - 10px) 100%, 0 100%, 0 10px)",
                        color: wakeListening ? "#00ffe1" : "#5a8a9f",
                        fontSize: 12, letterSpacing: "0.1em", cursor: "pointer",
                        backdropFilter: "blur(8px)", transition: "all 0.3s",
                        fontFamily: "'Rajdhani', sans-serif", fontWeight: 600,
                    }}
                >
                    <span style={{
                        width: 7, height: 7, borderRadius: "50%",
                        background: wakeDetected ? "#fff" : wakeListening ? "#00ffe1" : "#5a8a9f",
                        boxShadow: wakeListening ? "0 0 8px #00ffe1" : "none",
                        animation: wakeListening ? "pulse 1.5s infinite" : "none",
                        transition: "all 0.2s",
                    }} />
                    {wakeDetected ? "JARVIS ACTIVATED" : wakeListening ? "WAKE WORD ON" : "WAKE WORD OFF"}
                </button>
            </div>

            {isDraggingMode && (
                <div style={{ position: "absolute", top: "20%", pointerEvents: "none", color: "#00ffe1", letterSpacing: "2px", fontWeight: "bold", fontSize: "1.2rem", textShadow: "0 0 10px #00ffe1" }}>
                    DRAG THE BLOB TO REPOSITION
                </div>
            )}
            
            {listening && (
                <div style={{
                    position: "fixed",
                    bottom: 30,
                    right: 30,
                    width: "420px",
                    maxHeight: "300px",
                    padding: "20px 25px",
                    background: "rgba(0, 15, 25, 0.65)",
                    backdropFilter: "blur(20px)",
                    WebkitBackdropFilter: "blur(20px)",
                    borderBottom: "2px solid rgba(0, 255, 225, 0.4)",
                    borderLeft: "1px solid rgba(0, 255, 225, 0.2)",
                    clipPath: "polygon(20px 0, 100% 0, calc(100% - 20px) 100%, 0 100%, 0 20px)",
                    boxShadow: "0 20px 50px rgba(0,0,0,0.5), inset 0 0 20px rgba(0, 255, 225, 0.05)",
                    zIndex: 9000,
                    display: "flex",
                    flexDirection: "column",
                    gap: "15px",
                    overflowY: "auto"
                }}>
                    {/* User Text Section */}
                    <div>
                        <div style={{ color: "#00ffe1", fontSize: 10, letterSpacing: "2px", opacity: 0.7, marginBottom: 8 }}>▸ USER_INPUT</div>
                        <div style={{ color: "rgba(255,255,255,0.9)", fontSize: "16px", lineHeight: "1.4" }}>
                            {transcript ? (
                                <>
                                    <span style={{ color: "#00ffe1", marginRight: "8px" }}>›</span>
                                    <span>{finalTranscript}</span>
                                    {interimTranscript && <span style={{ opacity: 0.5, fontStyle: 'italic' }}>{interimTranscript}</span>}
                                </>
                            ) : (
                                <span style={{ opacity: 0.3, fontSize: 14 }}>Listening...</span>
                            )}
                        </div>
                    </div>

                    {/* Divider */}
                    <div style={{ height: "1px", background: "rgba(0, 255, 225, 0.15)", width: "100%" }} />

                    {/* JARVIS Response Section */}
                    <div style={{ opacity: jarvisResponse || isThinking ? 1 : 0, transition: "all 0.5s" }}>
                        <div style={{ color: "#ffaa00", fontSize: 10, letterSpacing: "2px", opacity: 0.7, marginBottom: 8 }}>
                            {isThinking ? "▸ ANALYZING..." : "▸ J.A.R.V.I.S"}
                        </div>
                        <div style={{ 
                            color: isThinking ? "#00ffe1" : "#fff", 
                            fontSize: "16px", 
                            lineHeight: "1.5", 
                            fontFamily: "'Share Tech Mono', monospace",
                            textShadow: isThinking ? "0 0 8px rgba(0,255,225,0.5)" : "none"
                        }}>
                            {jarvisResponse}
                            {isThinking && <span className="thinking-cursor" style={{ marginLeft: 5 }}>_</span>}
                        </div>
                    </div>
                </div>
            )}

            <div style={{ position: "fixed", bottom: 28, left: "50%", transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 10, zIndex: 500 }}>
                <button
                    onClick={toggleMic}
                    style={{
                        padding: "11px 30px",
                        borderRadius: 999,
                        border: listening ? "1.5px solid #00ffe1" : "1.5px solid #0066ff",
                        background: listening ? "rgba(0,255,225,0.08)" : "rgba(0,102,255,0.08)",
                        color: listening ? "#00ffe1" : "#4da6ff",
                        fontSize: 14,
                        letterSpacing: "0.06em",
                        cursor: "pointer",
                        transition: "all 0.3s",
                        backdropFilter: "blur(8px)",
                        fontFamily: "'Rajdhani', sans-serif",
                        fontWeight: 600,
                    }}
                >
                    {listening ? "⏹  STOP" : "🎙  START LISTENING"}
                </button>
                {permissionDenied && (
                    <span style={{ color: "#ff4466", fontSize: 12 }}>Microphone access denied</span>
                )}
            </div>
        </div>
    );
}
