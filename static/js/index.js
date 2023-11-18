$(document).ready(function() {
    function appendMessage(sender, message) {
        $("#chat-display").append("<p><strong>" + sender + ": </strong>" + message + "</p>");
    }

    function sendUserInput() {
        var userInput = $("#user-input").val();
        appendMessage("You", userInput);
        $("#user-input").val("");

        $.post("/ask", { user_input: userInput }, function(data) {
            var response = data.response;
            appendMessage("Chatbot", response);
            responsiveVoice.speak(response);
        });
    }

    $("#send-btn").click(sendUserInput);
    $("#user-input").keypress(function(e) {
        if (e.which === 13) {
            sendUserInput();
        }
    });
});


