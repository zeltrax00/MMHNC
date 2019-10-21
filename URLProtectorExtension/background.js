function listener(requestDetails) {
    console.log("Loading: " + requestDetails.url);
    
    let safe = false;
    
    // Check các kiểu rồi gán kết quả cho biến safe
    
    if (safe)
        return {cancel: false};
    else
    {
        let c = confirm("Trang web có dấu hiệu không an toàn, bạn có muốn tiếp tục ?");
        
        if (c == true)
            return {cancel: false};
        else
            return {cancel: true};
    }
}

chrome.webRequest.onBeforeRequest.addListener(
  listener,
  {
      urls: ['<all_urls>'], 
      types: ["main_frame"],
  },
  ['blocking']
);

