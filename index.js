async function fetchingAPI() {
  const res = await fetch("https://jsonplaceholder.typicode.com/todos/1");
  const response = await res.json();
  console.log(response);
}
function printConsole() {
  console.log("test test");
}
fetchingAPI();
printConsole();
printConsole();
printConsole();
printConsole();
printConsole();
printConsole();
printConsole();
printConsole();
