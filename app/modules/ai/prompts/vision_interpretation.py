"""
Prompts for interpreting vision predictions in Turkish.
"""
from app.modules.ai.prompts.templates import PromptTemplate

VISION_INTERPRETATION_TR = PromptTemplate(
    name="vision_interpretation_tr",
    template=(
        "Aşağıda HantaProject görüntü sınıflandırıcısının çıktısı yer alıyor. "
        "Bu sonucu Türkçe, sade ve kısa bir dille kullanıcıya açıkla.\n"
        "\n"
        "Kabul edildi mi: {accepted}\n"
        "Tahmin edilen sınıf: {predicted_class}\n"
        "Güven skoru: {confidence}\n"
        "Eşik değeri: {threshold}\n"
        "Reddedilme nedeni: {rejection_reason}\n"
        "Görüntü ilgililik kapısı (gate) etkin mi: {gate_enabled}\n"
        "Görüntü ilgililik kapısı sınıfı: {gate_predicted_class}\n"
        "Kullanılan model: {model_name} (sürüm: {model_version})\n"
        "\n"
        "Açıklamada şunlara dikkat et:\n"
        "1. Reddedildiyse nedenini sade dille söyle, kullanıcıya ne yapması "
        "gerektiğini öner (örn. daha net görüntü yükleme).\n"
        "2. Kabul edildiyse güven skorunun ne anlama geldiğini açıkla.\n"
        "3. Görüntü tabanlı tahminin tıbbi teşhis yerine geçmediğini hatırlat.\n"
        "Yanıt 6 cümleyi geçmesin."
    ),
)