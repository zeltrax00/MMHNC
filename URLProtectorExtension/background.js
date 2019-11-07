
function listener(requestDetails) {
    console.log("Loading: " + String(requestDetails.url));
    let safe = "true";
    // Check các kiểu rồi gán kết quả cho biến safe
    var req = new XMLHttpRequest();
    req.open('GET', 'http://localhost:5000/?url=' + requestDetails.url, false);
    req.send(null);
    safe = req.responseText;
    if (safe == "true")
        return {cancel: false};
    else if (safe == "false")
    {
        let c = confirm("Trang web " + requestDetails.url + " có dấu hiệu không an toàn, bạn có muốn tiếp tục ?");
        if (c == true)
            return {cancel: false};
        else
            return {cancel: true};
    }
    else
        console.log("Error server: " + req.responseText);
}

chrome.webRequest.onBeforeRequest.addListener(
  listener,
  {
      urls: ['<all_urls>'], 
      types: ["main_frame"],
  },
  ['blocking']
);

