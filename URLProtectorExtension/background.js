
function listener(requestDetails) {
    console.log("Loading: " + String(requestDetails.url));
    let safe = "true";
    // Check các kiểu rồi gán kết quả cho biến safe
    var req = new XMLHttpRequest();

    req.open('GET', 'http://localhost:5000/', false);

    safe = req.send(requestDetails.url);

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
        console.log("Error from server: " + safe);
}

chrome.webRequest.onBeforeRequest.addListener(
  listener,
  {
      urls: ['<all_urls>'], 
      types: ["main_frame"],
  },
  ['blocking']
);

