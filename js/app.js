'use strict';

/*
 * A Design by GraphBerry
 * Author: GraphBerry
 * Author URL: http://graphberry.com
 * License: http://graphberry.com/pages/license
 */
 jQuery(document).ready(function ($) {

    var lastId,
    topMenu = $("#top-navigation"),
    topMenuHeight = topMenu.outerHeight(),
        // All list items
        menuItems = topMenu.find("a"),
        // Anchors corresponding to menu items
        scrollItems = menuItems.map(function () {
            var href = $(this).attr("href");
            if(href.indexOf("#") === 0){
                var item = $($(this).attr("href"));
                if (item.length) {
                    return item;
                }
            }
        });

    //Get width of container
    var containerWidth = $('.section .container').width();
    //Resize animated triangle
    $(".triangle").css({
        "border-left": containerWidth / 2 + 'px outset transparent',
        "border-right": containerWidth / 2 + 'px outset transparent'
    });
    $(window).resize(function () {
        containerWidth = $('.container').width();
        $(".triangle").css({
            "border-left": containerWidth / 2 + 'px outset transparent',
            "border-right": containerWidth / 2 + 'px outset transparent'
        });
    });


    //Initialize header slider.
    $('#da-slider').cslider();

    // Bind to scroll
    $(window).scroll(function () {

        //Display or hide scroll to top button 
        if ($(this).scrollTop() > 100) {
            $('.scrollup').fadeIn();
        } else {
            $('.scrollup').fadeOut();
        }
    });

    /*
    Function for scroliing to top
    ************************************/
    $('.scrollup').click(function () {
        $("html, body").animate({
            scrollTop: 0
        }, 600);
        return false;
    });

    // Add smooth scrolling on all links inside the navbar
    $("#nav_menu a").on('click', function(event) {

        // Make sure this.hash has a value before overriding default behavior
        if (this.hash !== "") {
    
        // Prevent default anchor click behavior
        event.preventDefault();
    
        // Store hash
        var hash = this.hash;
    
        // Using jQuery's animate() method to add smooth page scroll
        // The optional number (800) specifies the number of milliseconds it takes to scroll to the specified area
        $('html, body').animate({
            scrollTop: $(hash).offset().top
        }, 800, function(){
    
        // Add hash (#) to URL when done scrolling (default click behavior)
            window.location.hash = hash;
        });
    
        } // End if
  
    });    

    //Function for show or hide portfolio desctiption.
    $.fn.showHide = function (options) {
        var defaults = {
            speed: 1000,
            easing: '',
            changeText: 0,
            showText: 'Show',
            hideText: 'Hide'
        };
        var options = $.extend(defaults, options);
        $(this).click(function () {
            $('.toggleDiv').slideUp(options.speed, options.easing);
            var toggleClick = $(this);
            var toggleDiv = $(this).attr('rel');
            $(toggleDiv).slideToggle(options.speed, options.easing, function () {
                if (options.changeText == 1) {
                    $(toggleDiv).is(":visible") ? toggleClick.text(options.hideText) : toggleClick.text(options.showText);
                }
            });
            return false;
        });
    };

    //Initial Show/Hide portfolio element.
    $('div.toggleDiv').hide();
    $('.show_hide').showHide({
        speed: 500,
        changeText: 0,
        showText: 'View',
        hideText: 'Close'
    });

    /************************
    Animate elements
    *************************/
    
    //Animate thumbnails 
    jQuery('.thumbnail').one('inview', function (event, visible) {
        if (visible == true) {
            jQuery(this).addClass("animated fadeInDown");
        } else {
            jQuery(this).removeClass("animated fadeInDown");
        }
    });

    //Animate triangles
    jQuery('.triangle').on('inview', function (event, visible) {
        if (visible == true) {
            jQuery(this).addClass("animated fadeInDown");
        } else {
            jQuery(this).removeClass("animated fadeInDown");
        }
    });
    
    //Animate contact form
    jQuery('.contact-form').on('inview', function (event, visible) {
        if (visible == true) {
            jQuery('.contact-form').addClass("animated bounceIn");
        } else {
            jQuery('.contact-form').removeClass("animated bounceIn");
        }
    });

    //Animate skill bars
    jQuery('.skills > li > span').one('inview', function (event, visible) {
        if (visible == true) {
            jQuery(this).each(function () {
                jQuery(this).animate({
                    width: jQuery(this).attr('data-width')
                }, 3000);
            });
        }
    });
});