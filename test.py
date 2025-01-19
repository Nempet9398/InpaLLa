# import requests
# import os

# def test_segmentation(original_image_path, prompt):
#     """
#     Flask 서버의 /lisa 엔드포인트에 이미지와 프롬프트를 전송하여
#     세그멘테이션 결과를 테스트하는 함수

#     Args:
#         original_image_path (str): 원본 이미지 파일 경로
#         prompt (str): 세그멘테이션 프롬프트
#     """
#     url = 'http://localhost:5001/lisa'  # Flask 서버 URL (포트 5001)

#     # 파일과 데이터 준비
#     files = {
#         'image': open(original_image_path, 'rb')
#     }
#     data = {
#         'prompt': prompt
#     }

#     try:
#         # POST 요청 보내기
#         response = requests.post(url, files=files, data=data)

#         # 응답 확인
#         if response.status_code == 200:
#             result = response.json()
#             print("Segmentation 성공!")
#             print("Mask Paths:")
#             for path in result.get("mask_paths", []):
#                 print(f" - {path}")
#             print("Masked Image Paths:")
#             for path in result.get("masked_image_paths", []):
#                 print(f" - {path}")
#         else:
#             # 오류 메시지 출력
#             try:
#                 error_message = response.json().get('error', '알 수 없는 오류')
#             except ValueError:
#                 error_message = '알 수 없는 오류'
#             print(f"Segmentation 실패: {error_message}")

#     except Exception as e:
#         print(f"요청 중 오류 발생: {e}")

# if __name__ == "__main__":
#     # 테스트할 이미지 경로와 프롬프트 설정
#     original_image = 'cat.png'  # 원본 이미지 파일 경로
#     prompt_text = 'Segment the cat'  # 세그멘테이션 프롬프트

#     # 테스트 실행
#     test_segmentation(original_image, prompt_text)

# import requests

# def test_inpainting(original_image_path, mask_image_path, prompt, guidance_scale, step, output_path):
#     url = 'http://localhost:5002/flux'  # Flask 서버 URL

#     # 파일과 데이터 준비
#     files = {
#         'original_image': open(original_image_path, 'rb'),
#         'mask_image': open(mask_image_path, 'rb')
#     }
#     data = {
#         'prompt': prompt,
#         'guidance_scale': guidance_scale,
#         'step': step
#     }

#     # POST 요청 보내기
#     response = requests.post(url, files=files, data=data)

#     # 응답 처리
#     if response.status_code == 200:
#         # 응답으로 받은 이미지를 파일로 저장
#         with open(output_path, 'wb') as f:
#             f.write(response.content)
#         print(f"Inpainting 성공! 결과 이미지가 {output_path}에 저장되었습니다.")
#     else:
#         # 오류 메시지 출력
#         try:
#             error_message = response.json().get('error', '알 수 없는 오류')
#         except ValueError:
#             error_message = '알 수 없는 오류'
#         print(f"Inpainting 실패: {error_message}")

# if __name__ == "__main__":
#     # 테스트할 이미지 경로 설정
#     original_image = 'cat.png'  # 원본 이미지 경로
#     mask_image = './results/mask_0.jpg'        # 마스크 이미지 경로
#     prompt_text = 'dog, lookup, golden RETRIEVER, baby'           # 인페인팅 프롬프트
#     guidance_scale = 45                                    # guidance_scale 값
#     steps = 50                                        # step 값
#     output_image = './results'  # 결과 이미지 저장 경로

#     test_inpainting(original_image, mask_image, prompt_text, guidance_scale, steps, output_image)

import requests

def test_llm(image_path, prompt, output_path):
    """
    LLM API를 테스트하기 위한 함수
    - image_path: 테스트할 이미지 경로
    - prompt: 사용자 제공 프롬프트
    - output_path: API 결과를 저장할 텍스트 파일 경로
    """
    # url = 'http://localhost:5003/llm'  # LLM Flask 서버 URL
    url = 'http://213.173.98.197:12077/llm'
    try:
        # 파일과 데이터 준비
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'prompt': prompt}

            # POST 요청 보내기
            response = requests.post(url, files=files, data=data)

        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            print("LLM API 호출 성공!")
            print("응답 데이터:", result)

            # 결과를 파일로 저장
            with open(output_path, 'w') as f:
                f.write("Output Text:\n")
                f.write(result['output_text'] + "\n\n")
                f.write("Inpainting Prompt:\n")
                f.write(result['inpainting_prompt'] + "\n\n")
                f.write("Segmentation Prompt:\n")
                f.write(result['segmentation_prompt'] + "\n")

            print(f"결과가 {output_path}에 저장되었습니다.")
        else:
            # 오류 처리
            try:
                error_message = response.json().get('error', '알 수 없는 오류')
            except ValueError:
                error_message = '알 수 없는 오류'
            print(f"LLM API 호출 실패: {error_message}")

    except Exception as e:
        print(f"테스트 실패: {str(e)}")

if __name__ == "__main__":
    # 테스트할 이미지 및 설정
    image_path = 'cat.png'                      # 테스트 이미지 경로
    prompt_text = 'Change cat character to dog.'  # 테스트 프롬프트
    output_path = './llm_results.txt'          # 결과 저장 경로

    test_llm(image_path, prompt_text, output_path)