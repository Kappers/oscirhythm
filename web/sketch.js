/*
@author T. Kaplan
*/

// Performance may improve by disabling 'friendly error system'
p5.disableFriendlyErrors = true;

// Create a small buffer for window marginal, as some browsers may not like full screen
const WINDOW_SCALE = .99;

const BG_COLOR = 0;
const OSC_RCOLOR = 40;

// Useful to lower volume of Tonejs, if it is loud - if using isolating headphones!
//Tone.Master.volume.value -= 20;

// Create one synth for the metronome, and schedule it at 60pm - BPM is kept higher globally as
// this ensures changes in GrFNN are sonified faster (it is updated per measure).
const BPM = 240;
Tone.Transport.bpm.value = BPM;
const metroSynth  = new Tone.PluckSynth().toMaster();
const metroLoop = new Tone.Loop(function(time){
    metroSynth.triggerAttackRelease('G4', '32n', time);
}, "1n");

// Create poly synth for possible active oscillators at one moment, and track the global array of
// noteTime tuples that get sequenced in a given measure.
var noteTimes = [];
const synth = new Tone.PolySynth(10, Tone.Synth).toMaster();

// Track elapsed per draw loop for our oscillators - keeps oscillators phased properly
var prevTime;

// Oscillator array, for representing GrFNN state, allowing MAX_NUM_GRFNNS at a given time
const MAX_NUM_GRFNNS = 3;
var currGrFNN = 0;
var grfnns = []
// GrFNN parameters - gradient 250 oscillators from 0.25Hz to 4.0Hz
const NUM_OSC = 250.0;
const OSC_FROM_FREQ = 0.25;
const OSC_TO_FREQ = 4.0;
const OSC_XSTART = 50;
const OSC_Y = 150;
const OSC_SPACING = 5.0;
const GRFNN_Y_SPACING = 120;

// If active oscillators drift by this amount in the GrFNN, it'll be treated as the same tone
const OSC_DIST_DRIFT = 20;

// Socket to receive GrFNN data from Flask
var socket = io.connect('http://127.0.0.1:5000/')

// Keys for data retrieved over sockets re GrFNN state, think (GrFNNKey from client.py)
const GRFNN_RESET_KEY = "GrFNN_Reset";
const GRFNN_DATA_KEY = "GrFNN_Data";
const GRFNN_AMPS_KEY = "GrFNN_Amps";
const GRFNN_PEAKS_KEY = "GrFNN_Peaks";
const GRFNN_ACTION_KEY = "GrFNN_Action";
// Action payloads expected with an ACTION_KEY
const ACTION_RESET = "RESET";
const ACTION_LOCK = "LOCK";
const ACTION_NEW_TONES = "NEW_TONES";
const ACTION_NEW_GRFNN = "NEW_GRFNN";

// Connection relay, for sanity on server-side logging
socket.on('connect', function() {
    socket.emit('connected', {data: 'JS Connected!'});
});

// When receiving updated GrFNN data, update the active model
socket.on(GRFNN_DATA_KEY, function(data) {
    if (!activeGrFNN().locked) {
        activeGrFNN().updateAmplitudes(data[GRFNN_AMPS_KEY], new Set(data[GRFNN_PEAKS_KEY]));
        scheduleOscNotes();
    }
});

// When receiving an action, switch and handle accordingly
socket.on(GRFNN_ACTION_KEY, function(action) {
    if (action === ACTION_RESET)
        reset();
    else if (action === ACTION_LOCK)
        activeGrFNN().lockToggle()
    else if (action === ACTION_NEW_TONES)
        changeTones();
    else if (action === ACTION_NEW_GRFNN)
        switchGrFNNLayer();
});

/* GrFNNLayer contains a single gradient of oscillators, and is capable of producing an independent
 * melody, recorded through the noteTimes and oscTones members 
 */
class GrFNNLayer {

    constructor(ind, num, fromFreq, toFreq, xStart, y, oscSpacing) {
        this.ind = ind;
        this.xStart = xStart;
        this.y = y;
        // The array of oscillators
        this.oscs = [];
        // The active oscillators, which will be sonified - these equate to peaks in a frequency
        // response magnitude spectrum.
        this.peakOscs = new Set();
        let freqDelta = (toFreq-fromFreq)/num;
        for (let i = 1; i <= num; i++) {
            let freq = fromFreq + freqDelta*i;
            this.oscs.push(new Oscillator(freq, xStart+(i*oscSpacing), y));
        }
        // Lock for rhythmic changes to GrFNN - user triggered
        this.locked = false;
        // Existing note mapping is used to ensure tonality doesn't jump around on oscillator drift
        this.oscTones = {};
        // (Note, Time) array to be merged with other layers and played in given measure
        this.noteTimes = [];
    }

    isActive() {
        return currGrFNN === this.ind;
    }

    lockToggle() {
        this.locked = !this.locked;
    }

    renderLabel() {
        let msg = "";
        let col = color(80);
        if (this.isActive()) {
            if (this.locked) {
                col = color(200, 0, 0);
                msg = "LOCKED";
            }
            else {
                col = color(0, 200, 100);
                msg = "ACTIVE";
            }
        }
        else {
            msg = "INACTIVE";
        }
        fill(col);
        // Rotate this label left by 90degs, to have it parallel with the left-most oscillator
        translate(this.xStart-20, this.y);
        rotate(-PI/2);
        textSize(17);
        text(msg, 0, 0);
        rotate(PI/2);
        translate(-(this.xStart-20), -this.y);
        noFill();
    }

    render(elapsed) {
        this.renderLabel();
        for (const osc of this.oscs) {
            osc.update(elapsed);
            osc.render(this.locked, this.isActive());
        }
    }

    changeTones() {
        Object.keys(this.oscTones).forEach(function(peak) {
            let tone = randomNote();
            while (tone === this.oscTones[peak])
                tone = randomNote();
            this.oscTones[peak] = tone;
        });
    }

    prepareOscNotes() {
        // Build a sorted mapping of peaks that are within OSC_DIST_DRIFT proximity - closePeaks
        let currTonalPeaks = Object.keys(this.oscTones);
        let closePeaks = [];
        let peakOscsList = Array.from(this.peakOscs);
        for (const peak of currTonalPeaks) {
            for (const newPeak of peakOscsList) {
                if (Math.abs(newPeak-peak) < OSC_DIST_DRIFT) {
                    let mapping = [newPeak, peak];
                    closePeaks.splice(getSortedIdxPairDiff(closePeaks, mapping), 0, mapping);
                }
            }
        }
        // Map the tones of drifted peaks accordingly, to avoid creating unwanted new tones
        let mappedTones = new Set([]);
        for (const closePeak of closePeaks) {
            let newPeak = closePeak[0];
            let existingPeak = closePeak[1];
            // If nothing has already mapped to this existing peak, take it
            if (!mappedTones.has(existingPeak)) {
                let tone = this.oscTones[existingPeak];
                delete this.oscTones[existingPeak];
                this.oscTones[newPeak] = tone;
            }
        }

        // Find the scheduled notes for each time delta
        let deltaMults = {};
        for (const peak of peakOscsList) {
            // Retrieve frequency of given 'peak' oscillator
            let dt = 1/this.oscs[peak].freq;
            // Create a random note, if a tone doesn't exist already for the oscillator
            let note = null;
            if (peak in this.oscTones)
                note = this.oscTones[peak];
            else {
                note = randomNote();
                this.oscTones[peak] = note;
            }
            // Find all time deltas within a measure for this oscillator - merging notes on a given
            // time delta in a set if shared with a different oscillator
            let t = 0.0;
            while (t < 4.0*1.0/(BPM/60.0)) {
                if (t in deltaMults)
                    deltaMults[t].add(note);
                else
                    deltaMults[t] = new Set([note]);
                t += dt;
            }
        }

        // Finally, switch out sequenced notes 
        let newNoteTimes = [];
        for (const dt in deltaMults)
            newNoteTimes.push([dt, deltaMults[dt]])
        this.noteTimes = newNoteTimes;
    }

    updateAmplitudes(amplitudeNorms, peakAmpNorms) {
        this.peakOscs = peakAmpNorms;
        for (let i = 0; i < this.oscs.length; i++)
            this.oscs[i].setAmplitude(amplitudeNorms[i], peakAmpNorms.has(i));
    }
}


class Oscillator {

    constructor(freq, x, y) {
        this.freq = freq;

        this.x = x;
        this.yOrigin = y;
        this.yDelta = 0.0;

        // Grow tracks whether the oscillator (visualised as a rectangle getting larger/smaller) is
        // currently growing or not (shrinking).
        this.grow = true;

        // Activated means this oscillator will be used to produce a tone, and it will be rendered
        // in a special way.
        this.activated = false;

        this.WIDTH = 1.0;
        this.W_MAX = 6.0;
        this.HEIGHT = 40.0;
        this.H_MAX = GRFNN_Y_SPACING;

        // Normalised amplitude threshold used to ensure not all slightly active oscillators are
        // rendered in a crazy colorful way.
        this.AMP_NORM_THRESH = 0.4;
        this.ampNorm = 0.0;
    }
    
    setAmplitude(normVal, activated) {
        this.activated = activated;
        this.ampNorm = normVal;
    }
    
    update(elapsed) {
        // Find the amount of phase change in the oscillator given time elapsed vs natural
        // frequency, use this to move grow or shrink the oscillator accordingly.
        let delta = elapsed/(1/this.freq/2.0);
        this.yDelta += this.grow ? delta : -delta;
        this.prevGrow = this.grow;
        if (this.grow && this.yDelta >= 1.0) {
            this.yDelta = 1.0-(this.yDelta-1.0);
            this.grow = !this.grow;
        }
        else if (!this.grow && this.yDelta <= 0.0) {
            this.yDelta = Math.abs(this.yDelta);
            this.grow = !this.grow;
        }
    }

    render(locked, activeLayer) {
        noStroke();
        let w = this.WIDTH;
        let h = this.HEIGHT;
        if (this.activated || (!locked && activeLayer)) {
            // Variable width/height boost as function of normalised amplitude.
            h += this.H_MAX*this.yDelta*max(this.ampNorm, this.AMP_NORM_THRESH);
            w += (this.W_MAX*this.yDelta*this.ampNorm);
        }

        let col = OSC_RCOLOR;
        // If the oscillator is active, make it bright red; and if it surpasses a norm. ampl. 
        // threshold, give it a duller red tint.
        if (this.activated)
            col = color(255, 0, 0);
        else if (!locked && activeLayer && this.ampNorm > this.AMP_NORM_THRESH) {
            let rVal = OSC_RCOLOR + this.ampNorm*30;
            col = color(rVal, 0, 0);
        }

        fill(col);
        rect(this.x, this.yOrigin, w, h);
        noFill();
    }
}

function activeGrFNN() {
    return grfnns[currGrFNN];
}

function reset() {
    let zeros = new Array(NUM_OSC+1).join('0.0').split('').map(parseFloat)
    activeGrFNN().updateAmplitudes(zeros, new Set([]));
    scheduleOscNotes();
}

function changeTones() {
    activeGrFNN().changeTones();
    scheduleOscNotes();
}

function switchGrFNNLayer() {
    if (!activeGrFNN().locked)
        activeGrFNN().lockToggle();
    currGrFNN = (currGrFNN+1) % grfnns.length;
}

function randomNote() {
    // TODO: Try and banish this rather crude method and assign tones systematically/sensibly.
    let notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
    let randNote = notes[Math.floor(Math.random() * notes.length)];
    let level =  Math.floor(Math.random() * 3) + 3.0;
    return randNote + level;
}
 
function zip(arrays) {
    // This trick is thanks to ninjagecko on https://stackoverflow.com/a/10284006
    return arrays[0].map(function(_,i){
        return arrays.map(function(array){return array[i]})
    });
}


function getSortedIdxPairDiff(pairs, tuple) {
    var low = 0,
    high = pairs.length;
    while (low < high) {
        var mid = low + high >>> 1;
        if (Math.abs(pairs[mid][1]-pairs[mid][0]) < Math.abs(tuple[1]-tuple[0]))
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

function scheduleOscNotes() {
    // If this is being called, there has likely been a model change - absorb it into active GrFNN
    activeGrFNN().prepareOscNotes();

    // Merge all note times across all GrFNN
    let newNoteTimes = [];
    for (let i = 0; i < MAX_NUM_GRFNNS; i++)
        newNoteTimes = newNoteTimes.concat(grfnns[i].noteTimes)
    // Sort the notetimes by time, which is unintuitively the first of each tuple
    newNoteTimes.sort(function(a, b) {
        return float(a[0]) -float( b[0])
    });

    // Merge notetimes where time offsets are the same
    // TODO: The management of different types here needs work - due to how things are generated
    // in each GrFNNLayer, then sorted above, there is redundant set<->array<->singleton conversion
    let mergedNoteTimes = [];
    if (newNoteTimes.length > 0) {
        mergedNoteTimes.push(newNoteTimes[0]);
        for (let j = 1; j < newNoteTimes.length; j++) {
            let lastInMerged = mergedNoteTimes.length-1;
            // If this time offset equals the current notetime on the mergedNoteTimes stack
            if (newNoteTimes[j][0] === mergedNoteTimes[lastInMerged][0]) {
                // Make sure everything is a set-y, be it the existing stack item or new chord/note
                if (Array.isArray(mergedNoteTimes[lastInMerged][1]))
                    mergedNoteTimes[lastInMerged][1] = new Set(mergedNoteTimes[lastInMerged][1]);
                else if (!(mergedNoteTimes[lastInMerged][1] instanceof Set))
                    mergedNoteTimes[lastInMerged][1] = new Set([mergedNoteTimes[lastInMerged][1]]);
                if (newNoteTimes[j][1] instanceof Set) {
                    for (const note of newNoteTimes[j][1])
                        mergedNoteTimes[lastInMerged][1].add(note)
                }
                else
                    mergedNoteTimes[lastInMerged][1].add(newNoteTimes[j][1])
            }
            else
                mergedNoteTimes.push(newNoteTimes[j]);
        }
    }

    // Ensure the notes are now an array, instead of set, to avoid WebAudio complaints (sigh)
    for (let i = 0; i < mergedNoteTimes.length; i++) {
        if (mergedNoteTimes[i][1] instanceof Set) {
            if (mergedNoteTimes[i][1].size > 1)
                mergedNoteTimes[i][1] = Array.from(mergedNoteTimes[i][1]);
            else
                mergedNoteTimes[i][1] = Array.from(mergedNoteTimes[i][1])[0];
        }
    }

    // Finally, switch out sequenced notes to be put in place for next measure!
    noteTimes = mergedNoteTimes;
}

// Every measure, schedule notes at respective times as per scheduleOscNotes - this allows notes
// and times to be rescheduled elsewhere without clogging the Transport
Tone.Transport.scheduleRepeat(function(time) {
    for (let i = 0; i < noteTimes.length; i++)
        synth.triggerAttackRelease(noteTimes[i][1], '32n', '+'+noteTimes[i][0]);
}, '1m');
Tone.Transport.loopEnd = '1m'
Tone.Transport.loop = true

function setup() {
    textAlign(CENTER, CENTER);
    // Create canvas using WINDOW_SCALE (e.g. .95) to give buffer around p5.js X and Y
    createCanvas($(window).width()*WINDOW_SCALE, $(window).height()*WINDOW_SCALE);
    background(BG_COLOR)
    rectMode(CENTER);

    // Create the different GrFNNLayers
    for (let i = 0; i < MAX_NUM_GRFNNS; i++) {
        let yOffset = i*GRFNN_Y_SPACING*1.4;
        let grfnn = new GrFNNLayer(i, NUM_OSC, OSC_FROM_FREQ, OSC_TO_FREQ, OSC_XSTART,
                                   OSC_Y+yOffset, OSC_SPACING);
        grfnn.render(0.0);
        grfnns.push(grfnn);
    }

    // Create a metronome, and sync it to the oscillator rendering (phase relies on prevTime)
    // Roughly align them to 'start' at some near future time, and add 100millis delay on Tonejs
    // as this is advised best practice to avoid scheduling errors.
    metroLoop.start(0);
    prevTime = Tone.now()+1.0;
    Tone.Transport.start(prevTime+0.1);
}

function draw() {
    // Each draw() is simple, find time elapsed since last and use that to manage oscillator
    // phase-sensitive rendering
    let currTime = Tone.now();
    var elapsed = currTime - prevTime;
    background(BG_COLOR);
    grfnns.forEach(function(grfnn) {
        grfnn.render(elapsed);
    });
    prevTime = currTime;
}
