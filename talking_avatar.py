import pygame
import sys
import numpy as np
import threading
import pyaudio
import audioop
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Elegant Talking Avatar")

# Colors
BACKGROUND = (25, 25, 40)
SKIN_LIGHT = (255, 214, 175)
SKIN_MEDIUM = (245, 194, 150)
SKIN_DARK = (235, 174, 125)
LIP_COLOR = (200, 90, 90)
LIP_INNER = (220, 130, 130)
EYE_COLOR = (250, 250, 255)
PUPIL_COLOR = (60, 50, 90)
EYESHADOW = (180, 130, 160)
HAIR_COLOR = (40, 25, 15)
BLUSH_COLOR = (255, 180, 180)
SHIRT_COLOR = (80, 100, 160)
NECKLACE_COLOR = (220, 190, 100)
UI_BG = (35, 35, 55)
UI_ACCENT = (110, 170, 220)

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Mouth animation parameters
mouth_openness = 0
volume_threshold = 500
smoothing_factor = 0.2

# Font
title_font = pygame.font.SysFont("Arial", 36, bold=True)
font = pygame.font.SysFont("Arial", 20)
small_font = pygame.font.SysFont("Arial", 16)

# Avatar state
talking = False
volume_level = 0
blink_timer = 0
blink_interval = 120  # frames between blinks
blinking = False
blink_duration = 0

# Audio processing function
def process_audio():
    global mouth_openness, talking, volume_level
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            volume = audioop.rms(data, 2)
            volume_level = volume
            
            # Smooth the mouth animation
            target_openness = min(volume / 3000, 1.0) if volume > volume_threshold else 0
            mouth_openness = mouth_openness * (1 - smoothing_factor) + target_openness * smoothing_factor
            
            talking = volume > volume_threshold
        except Exception as e:
            print(f"Audio processing error: {e}")
            break

# Start audio processing in a separate thread
audio_thread = threading.Thread(target=process_audio, daemon=True)
audio_thread.start()

# Draw the elegant avatar
def draw_avatar():
    # Head base
    pygame.draw.ellipse(screen, SKIN_LIGHT, (WIDTH//2 - 140, HEIGHT//2 - 220, 280, 320))
    
    # Neck
    pygame.draw.rect(screen, SKIN_MEDIUM, (WIDTH//2 - 50, HEIGHT//2 + 100, 100, 60))
    
    # Hair (behind head)
    pygame.draw.ellipse(screen, HAIR_COLOR, (WIDTH//2 - 150, HEIGHT//2 - 240, 300, 200))
    pygame.draw.ellipse(screen, HAIR_COLOR, (WIDTH//2 - 160, HEIGHT//2 - 180, 320, 280))
    
    # Shoulders
    pygame.draw.ellipse(screen, SHIRT_COLOR, (WIDTH//2 - 200, HEIGHT//2 + 150, 400, 120))
    
    # Blush (subtle)
    if mouth_openness > 0.2:
        for x_offset in [-80, 80]:
            blush_alpha = min(150, int(mouth_openness * 200))
            blush_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
            pygame.draw.ellipse(blush_surf, (*BLUSH_COLOR, blush_alpha), (0, 0, 40, 20))
            screen.blit(blush_surf, (WIDTH//2 + x_offset - 20, HEIGHT//2 + 20))
    
    # Eyes
    blink_progress = min(1.0, blink_duration / 10) if blinking else 0
    eye_openness = 1.0 - abs(math.sin(blink_progress * math.pi))
    
    for x_offset in [-60, 60]:
        # Eyeshadow
        pygame.draw.ellipse(screen, EYESHADOW, (WIDTH//2 + x_offset - 35, HEIGHT//2 - 50, 70, 40 * eye_openness))
        
        # Eye whites
        pygame.draw.ellipse(screen, EYE_COLOR, (WIDTH//2 + x_offset - 30, HEIGHT//2 - 45, 60, 30 * eye_openness))
        
        # Pupils (move slightly based on talking state)
        pupil_offset_x = 5 if talking else 0
        pupil_offset_y = -3 if mouth_openness > 0.5 else 0
        
        pygame.draw.circle(screen, PUPIL_COLOR, 
                          (WIDTH//2 + x_offset + pupil_offset_x, 
                           HEIGHT//2 - 30 + pupil_offset_y), 
                          10 * eye_openness)
        
        # Eyelashes
        for lash_offset in [-20, -10, 0, 10, 20]:
            lash_x = WIDTH//2 + x_offset + lash_offset
            lash_start_y = HEIGHT//2 - 45
            lash_end_y = HEIGHT//2 - 55 - (5 * (1 - eye_openness))
            pygame.draw.line(screen, HAIR_COLOR, (lash_x, lash_start_y), (lash_x, lash_end_y), 2)
    
    # Eyebrows
    brow_raise = 5 * mouth_openness
    for x_offset in [-60, 60]:
        pygame.draw.ellipse(screen, HAIR_COLOR, (WIDTH//2 + x_offset - 30, HEIGHT//2 - 80 - brow_raise, 60, 15))
    
    # Nose
    pygame.draw.polygon(screen, SKIN_DARK, [
        (WIDTH//2, HEIGHT//2 - 20),
        (WIDTH//2 - 15, HEIGHT//2 + 30),
        (WIDTH//2 + 15, HEIGHT//2 + 30)
    ])
    
    # Mouth - the main animated part
    mouth_width = 60 + mouth_openness * 40
    mouth_height = 10 + mouth_openness * 50
    
    # Outer lips
    pygame.draw.ellipse(screen, LIP_COLOR, (
        WIDTH//2 - mouth_width//2,
        HEIGHT//2 + 60 - mouth_height//2,
        mouth_width,
        mouth_height
    ))
    
    # Inner mouth (visible when open)
    if mouth_openness > 0.3:
        inner_height = max(5, mouth_height * 0.7)
        pygame.draw.ellipse(screen, (50, 30, 40), (
            WIDTH//2 - mouth_width*0.8//2,
            HEIGHT//2 + 60 - inner_height//2,
            mouth_width * 0.8,
            inner_height
        ))
        
        # Teeth (subtle)
        teeth_width = mouth_width * 0.7
        teeth_height = inner_height * 0.6
        pygame.draw.ellipse(screen, (250, 240, 230), (
            WIDTH//2 - teeth_width//2,
            HEIGHT//2 + 60 - teeth_height//2,
            teeth_width,
            teeth_height
        ))
    
    # Lip shine (subtle highlight)
    if mouth_openness < 0.7:
        highlight_width = mouth_width * 0.4
        highlight_height = mouth_height * 0.2
        pygame.draw.ellipse(screen, (255, 255, 255, 100), (
            WIDTH//2 - highlight_width//2,
            HEIGHT//2 + 60 - mouth_height//2 + 5,
            highlight_width,
            highlight_height
        ))
    
    # Ear
    pygame.draw.ellipse(screen, SKIN_MEDIUM, (WIDTH//2 + 140, HEIGHT//2 - 30, 40, 60))
    pygame.draw.ellipse(screen, SKIN_LIGHT, (WIDTH//2 + 145, HEIGHT//2 - 20, 30, 40))
    
    # Necklace
    pygame.draw.ellipse(screen, NECKLACE_COLOR, (WIDTH//2 - 40, HEIGHT//2 + 130, 80, 20))
    pygame.draw.circle(screen, NECKLACE_COLOR, (WIDTH//2, HEIGHT//2 + 140), 8)
    
    # Hair strands (over head)
    for i in range(-100, 100, 20):
        strand_x = WIDTH//2 + i
        if abs(i) < 80:
            pygame.draw.arc(screen, HAIR_COLOR, 
                           (strand_x - 20, HEIGHT//2 - 250, 40, 100),
                           math.pi, 2*math.pi, 2)
    
    # Earring
    pygame.draw.circle(screen, (220, 190, 100), (WIDTH//2 + 160, HEIGHT//2 + 5), 5)

# Draw UI elements
def draw_ui():
    # Draw title
    title = title_font.render("Elegant Talking Avatar", True, (220, 220, 240))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    # Draw volume meter background
    pygame.draw.rect(screen, UI_BG, (WIDTH // 2 - 150, HEIGHT - 120, 300, 25), border_radius=5)
    
    # Fill volume meter based on volume level
    meter_width = min(volume_level / 5000 * 300, 300)
    pygame.draw.rect(screen, UI_ACCENT, (WIDTH // 2 - 150, HEIGHT - 120, meter_width, 25), border_radius=5)
    
    # Draw volume meter border
    pygame.draw.rect(screen, (200, 200, 200), (WIDTH // 2 - 150, HEIGHT - 120, 300, 25), 2, border_radius=5)
    
    # Draw volume text
    volume_text = font.render(f"Volume: {volume_level}", True, (220, 220, 240))
    screen.blit(volume_text, (WIDTH // 2 - volume_text.get_width() // 2, HEIGHT - 90))
    
    # Draw status with color change when talking
    status_color = (150, 240, 150) if talking else (220, 220, 240)
    status_text = font.render(f"Status: {'TALKING' if talking else 'listening...'}", True, status_color)
    screen.blit(status_text, (WIDTH // 2 - status_text.get_width() // 2, HEIGHT - 60))
    
    # Draw instructions
    instructions = font.render("Speak into your microphone to make the avatar talk", True, (180, 180, 200))
    screen.blit(instructions, (WIDTH // 2 - instructions.get_width() // 2, HEIGHT - 160))
    
    # Draw decorative elements
    for i in range(0, WIDTH, 30):
        pygame.draw.line(screen, (60, 60, 80), (i, 0), (i, HEIGHT), 1)
    
    for i in range(0, HEIGHT, 30):
        pygame.draw.line(screen, (60, 60, 80), (0, i), (WIDTH, i), 1)
    
    # Draw footer
    footer_text = small_font.render("Voice-Controlled Avatar Demo • Made with PyGame", True, (150, 150, 170))
    screen.blit(footer_text, (WIDTH // 2 - footer_text.get_width() // 2, HEIGHT - 30))

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Handle blinking
    blink_timer += 1
    if blinking:
        blink_duration += 1
        if blink_duration > 10:
            blinking = False
            blink_duration = 0
    elif blink_timer > blink_interval:
        blinking = True
        blink_timer = 0
    
    # Clear the screen
    screen.fill(BACKGROUND)
    
    # Draw the avatar
    draw_avatar()
    
    # Draw UI
    draw_ui()
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

# Clean up
pygame.quit()
sys.exit()