function recognize() {
    let modelType = document.getElementById("inputGroupSelect01").value;
    let blobImage = $('#inputGroupFile04').get(0).files[0];
    let formData = new FormData();

    formData.append("image", blobImage);

    $.ajax({
        url: "/recognize/" + modelType,
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            window.location.replace(response);
        },
        error: function (jqXHR, textStatus, errorMessage) {
            console.log(errorMessage); // Optional
        }
    });
}

function preview(input) {
    if (input.files && input.files[0]) {
        let reader = new FileReader();
        let size = input.files[0].size;
        reader.onload = function (e) {
            $('#imagePreview')
                .attr('src', e.target.result)
                .width(size[0])
                .height(size[1]);
            document.getElementById("imagePreview").style.display = "block";
            document.getElementsByClassName("custom-file-label")[0].innerHTML = input.files[0].name;
        };
        reader.readAsDataURL(input.files[0]);
    }
}