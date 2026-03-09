// A quick check to see if we can find other fetch calls
const fs = require('fs');
const html = fs.readFileSync('f1pred/templates/index.html', 'utf8');
const lines = html.split('\n');
for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('fetch(') || lines[i].includes('.json()')) {
        console.log(`Line ${i+1}: ${lines[i].trim()}`);
    }
}
