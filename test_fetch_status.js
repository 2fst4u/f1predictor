// Simulating the fetch calls that might return 500
async function test() {
    let r = await fetch("http://127.0.0.1:8000/api/event-status/invalid/1");
    let t = await r.json();
    console.log(t);
}
test()
