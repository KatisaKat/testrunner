function isInViewport(element, threshold = 0.1) {
    const rect = element.getBoundingClientRect();
    const elementHeight = rect.height;

    // Calculate how much of the element is visible, based on the threshold percentage
    const visibleHeight = Math.min(Math.max(0, window.innerHeight - rect.top), elementHeight);
    
    // Check if the visible area exceeds the threshold percentage
    return visibleHeight / elementHeight >= threshold;
}

window.addEventListener('scroll', function (event) {
    const hiddenElements = document.querySelectorAll('.hidden, .hiddenright, .hiddenleft');

    hiddenElements.forEach((el) => {
        if (isInViewport(el, 0.25)) {  // Set threshold to 25% (you can change this value)
            el.classList.add('show');
        } else {
            el.classList.remove('show');
        }
    });
}, false);