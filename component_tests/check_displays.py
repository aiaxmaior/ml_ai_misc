import pygame
import time

print("Initializing Pygame to check displays...")

try:
    pygame.init()

    num_displays = pygame.display.get_num_displays()
    print(f"\nPygame has detected {num_displays} display(s).")
    print("-----------------------------------------")

    if num_displays > 0:
        for i in range(num_displays):
            try:
                # Get the size of the current display
                display_info = pygame.display.Info()
                w, h = display_info.current_w, display_info.current_h
                
                print(f"Now attempting to show a window on display index: {i}")

                # Create a small, temporary window on the current display index
                screen = pygame.display.set_mode(
                    (400, 200), pygame.NOFRAME, display=i
                )
                
                # Fill the window with a color and update the display
                screen.fill((255, 255, 255)) # White background
                
                font = pygame.font.Font(None, 74)
                text = font.render(f"Display Index: {i}", True, (0, 0, 0))
                text_rect = text.get_rect(center=(200, 100))
                screen.blit(text, text_rect)
                
                pygame.display.set_caption(f"Testing Display {i}")
                pygame.display.flip()
                
                # Keep the window open for a few seconds
                time.sleep(4)

            except pygame.error as e:
                print(f"Could not create a window on display {i}. Error: {e}")
            finally:
                # Close the window before opening the next one
                pygame.display.quit()
                pygame.init() # Re-initialize for the next loop iteration

    print("-----------------------------------------")
    print("Display check complete.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    pygame.quit()