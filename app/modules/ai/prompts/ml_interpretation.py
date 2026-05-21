"""
Prompts for interpreting tabular ML predictions in Turkish.
"""
from app.modules.ai.prompts.templates import PromptTemplate

ML_INTERPRETATION_TR = PromptTemplate(
    name="ml_interpretation_tr",
    template=(
        "Aşağıda hantavirüs risk modelinin bir hasta için ürettiği tahmin yer alıyor. "
        "Bunu Türkçe, sade ve kısa bir dille kullanıcıya açıkla.\n"
        "\n"
        "Tahmin etiketi: {label}\n"
        "Tahmin sınıfı (0=negatif, 1=pozitif): {prediction}\n"
        "Pozitif olasılığı: {probability}\n"
        "Güven seviyesi: {confidence}\n"
        "Kullanılan model: {model_name} (sürüm: {model_version})\n"
        "Hasta öne çıkan girdileri: {feature_summary}\n"
        "\n"
        "Açıklamada şunları içer:\n"
        "1. Tahminin ne anlama geldiği.\n"
        "2. Güven seviyesinin nasıl yorumlanması gerektiği.\n"
        "3. Sonucun kesin teşhis olmadığı, mutlaka bir hekime danışılması gerektiği uyarısı.\n"
        "Yanıt 6 cümleyi geçmesin."
    ),
)