$(document).ready(function () {
    "use strict";

    var $container = $('<div id="container"></div>');
    _.each(data, function (point) {
        var $point = $('<div class="point" title="' + point.label + '"></div>');
        $point.css({top: 100 * point.y + '%', left: 100 * point.x + '%'});
        $container.append($point);
    });
    $('body').append($container);

});
