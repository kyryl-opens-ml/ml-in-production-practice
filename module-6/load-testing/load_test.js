import http from 'k6/http';
import { sleep } from 'k6';

const movie_reviews = [
    "A rollercoaster of emotions with stunning visuals and remarkable performances. A must-see!",
    "Despite its high production values, the plot is predictable and lacks originality.",
    "An epic space opera that pulls you in with its intricate plot and complex characters.",
    "Too reliant on CGI, and the storyline feels disjointed and hard to follow.",
    "An extraordinary cinematic experience that beautifully captures the human spirit.",
    "The pacing is too slow, and it tends to feel more like a documentary than a feature film.",
    "A superb adaptation with a gripping plot and fascinating characters. Truly unforgettable.",
    "Though the scenery is beautiful, the characters feel flat and the storyline lacks depth.",
    "A touching story of love and loss, paired with phenomenal acting. It will leave you teary-eyed.",
    "The script is clichéd, and the chemistry between the lead actors feels forced.",
    "A thrilling and suspenseful journey that keeps you on the edge of your seat till the end.",
    "The plot twists feel contrived, and the horror elements seem more comical than scary.",
    "A poignant exploration of life and love, combined with a mesmerizing soundtrack.",
    "The narrative is overly sentimental and fails to deliver a strong message.",
    "An underwater adventure that's both visually stunning and emotionally resonant.",
    "The visual effects overshadow the story, which is lacking in depth and originality.",
    "An action-packed thrill ride with memorable characters and an engaging plot.",
    "The action scenes are overdone and the storyline is paper thin.",
    "A captivating sci-fi thriller that challenges your perception of reality.",
    "The plot is confusing and the ending leaves too many questions unanswered.",
];

export let options = {
    vus: 10,
    duration: '10m',
};

export default function () {
    sleep(Math.random() * 4 + 1);
    const num_of_review = Math.floor(Math.random() * 100) + 1;
    const reviews = [];
    for (let i = 0; i < num_of_review; i++) {
        const random_index = Math.floor(Math.random() * movie_reviews.length);
        reviews.push(movie_reviews[random_index]);
    }
    const payload = JSON.stringify({ text: reviews });
    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    http.post('http://0.0.0.0:8080/predict', payload, params);
}